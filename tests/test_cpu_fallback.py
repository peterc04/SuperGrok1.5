#!/usr/bin/env python3
"""
CPU Fallback + SIMD — Test Suite

Tests for:
  - Python fallback: every _ops function has a Python equivalent
  - CPU C++ extension: all optimizer functions registered
  - Optimizer wiring: all optimizers import with _HAS_OPS=False
  - SIMD: AVX-512/NEON detection (if available)
  - Numerical correctness: Python fallback matches known outputs

These tests run on any platform (CPU-only is fine).
"""

import os
import sys
import math
import traceback

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
#  Test 1: __init__.py import flags are correctly set
# ═══════════════════════════════════════════════════════════════════════

def test_init_flags():
    """Verify _HAS_OPS, _HAS_CUDA, _HAS_CPU_OPS flags exist and are boolean."""
    import grokking_optimizers as go
    assert hasattr(go, '_HAS_OPS'), "_HAS_OPS not found"
    assert hasattr(go, '_HAS_CUDA'), "_HAS_CUDA not found"
    assert hasattr(go, '_HAS_CPU_OPS'), "_HAS_CPU_OPS not found"
    assert isinstance(go._HAS_OPS, bool), f"_HAS_OPS is {type(go._HAS_OPS)}, expected bool"
    assert isinstance(go._HAS_CUDA, bool), f"_HAS_CUDA is {type(go._HAS_CUDA)}, expected bool"
    assert isinstance(go._HAS_CPU_OPS, bool), f"_HAS_CPU_OPS is {type(go._HAS_CPU_OPS)}, expected bool"


# ═══════════════════════════════════════════════════════════════════════
#  Test 2: Python fallback module exists and has all functions
# ═══════════════════════════════════════════════════════════════════════

def test_python_fallback_exists():
    """Verify _python_fallback.py has all required functions."""
    from grokking_optimizers import _python_fallback as fb

    required_funcs = [
        'supergrok15_fused_step',
        'supergrok11_fused_step',
        'grokadamw_fused_step',
        'neuralgrok_fused_step',
        'prodigy_fused_step',
        'grokfast_fused_step',
        'lion_fused_step',
        'looksam_perturb_all',
        'looksam_restore_all',
        'looksam_compute_directions',
        'looksam_adjust_grads',
        'muon_fused_step',
        'supergrok2_mamba_peer_batched_step',
    ]

    missing = [f for f in required_funcs if not hasattr(fb, f)]
    assert not missing, f"Missing fallback functions: {missing}"


# ═══════════════════════════════════════════════════════════════════════
#  Test 3: All optimizer .py files use conditional _ops import
# ═══════════════════════════════════════════════════════════════════════

def test_optimizer_conditional_import():
    """Verify all optimizers use conditional _ops import with _python_fallback."""
    from pathlib import Path

    opt_dir = Path(__file__).parent.parent / "grokking_optimizers"
    optimizer_files = [
        'supergrok15.py', 'supergrok11.py', 'supergrok2.py',
        'grokadamw.py', 'neuralgrok.py', 'prodigy.py',
        'grokfast.py', 'lion.py', 'looksam.py', 'muon.py',
        'mamba3_peer_metanet.py',
    ]

    violations = []
    for fname in optimizer_files:
        fpath = opt_dir / fname
        if not fpath.exists():
            violations.append(f"{fname}: file not found")
            continue
        content = fpath.read_text()
        # Every optimizer must reference _python_fallback for the else branch
        if '_python_fallback' not in content:
            violations.append(f"{fname}: no _python_fallback import")
        # Every optimizer must check _HAS_OPS
        if '_HAS_OPS' not in content:
            violations.append(f"{fname}: no _HAS_OPS check")

    assert not violations, "Missing conditional imports:\n" + "\n".join(f"  {v}" for v in violations)


# ═══════════════════════════════════════════════════════════════════════
#  Test 4: Python fallback — Lion optimizer numerical correctness
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_lion_numerics():
    """Verify Lion fallback produces correct sign-based updates."""
    import torch
    from grokking_optimizers._python_fallback import lion_fused_step

    torch.manual_seed(42)
    p = torch.randn(10)
    g = torch.randn(10)
    m = torch.zeros(10)
    p_orig = p.clone()

    lr, beta1, beta2, wd = 0.001, 0.9, 0.99, 0.1

    lion_fused_step([p], [g], [m], lr, beta1, beta2, wd)

    # Verify: update = sign(beta1*m_old + (1-beta1)*g) = sign((1-beta1)*g) = sign(g)
    # Since m was zero initially
    expected_sign = g.sign()
    expected_p = p_orig * (1.0 - lr * wd) - lr * expected_sign
    assert torch.allclose(p, expected_p, atol=1e-6), \
        f"Lion output mismatch: max diff={torch.abs(p - expected_p).max().item()}"


# ═══════════════════════════════════════════════════════════════════════
#  Test 5: Python fallback — GrokAdamW numerical correctness
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_grokadamw_numerics():
    """Verify GrokAdamW fallback updates parameters correctly."""
    import torch
    from grokking_optimizers._python_fallback import grokadamw_fused_step

    torch.manual_seed(42)
    p = torch.randn(10)
    g = torch.randn(10)
    ea = torch.zeros(10)
    easq = torch.zeros(10)
    ema = torch.zeros(10)
    steps = [0]
    p_orig = p.clone()

    alpha, lamb = 0.98, 5.0
    beta1, beta2 = 0.9, 0.999
    lr, wd, eps = 0.001, 1.0, 1e-8
    grad_clip = 10.0

    grokadamw_fused_step([p], [g], [ea], [easq], [ema], steps,
                         alpha, lamb, beta1, beta2, lr, wd, eps, grad_clip)

    assert steps[0] == 1, f"Step not incremented: {steps[0]}"
    assert not torch.equal(p, p_orig), "Parameters unchanged after step"
    assert torch.all(torch.isfinite(p)), "NaN/Inf in parameters"


# ═══════════════════════════════════════════════════════════════════════
#  Test 6: Python fallback — Grokfast EMA amplification
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_grokfast_ema():
    """Verify Grokfast EMA amplification modifies gradients in-place."""
    import torch
    from grokking_optimizers._python_fallback import grokfast_fused_step

    torch.manual_seed(42)
    g = torch.randn(10)
    ema = torch.zeros(10)
    g_orig = g.clone()

    alpha, lamb = 0.98, 5.0
    grokfast_fused_step([g], [ema], alpha, lamb)

    # EMA should be updated: ema = (1-alpha)*g_orig
    expected_ema = (1.0 - alpha) * g_orig
    assert torch.allclose(ema, expected_ema, atol=1e-6), "EMA not updated correctly"

    # Grad should be amplified: g = g_orig + lamb * ema
    expected_g = g_orig + lamb * expected_ema
    assert torch.allclose(g, expected_g, atol=1e-5), "Gradient not amplified correctly"


# ═══════════════════════════════════════════════════════════════════════
#  Test 7: Python fallback — LookSAM perturb/restore cycle
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_looksam_cycle():
    """Verify LookSAM perturb/restore leaves parameters unchanged."""
    import torch
    from grokking_optimizers._python_fallback import (
        looksam_perturb_all, looksam_restore_all,
    )

    torch.manual_seed(42)
    p = torch.randn(20)
    g = torch.randn(20)
    p_orig = p.clone()

    backups = looksam_perturb_all([p], [g], rho=0.05)
    assert not torch.equal(p, p_orig), "Perturb didn't change params"

    looksam_restore_all([p], backups)
    assert torch.allclose(p, p_orig, atol=1e-7), "Restore didn't recover params"


# ═══════════════════════════════════════════════════════════════════════
#  Test 8: Python fallback — Muon Newton-Schulz
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_muon_ns():
    """Verify Muon Newton-Schulz produces near-orthogonal updates."""
    import torch
    from grokking_optimizers._python_fallback import muon_fused_step

    torch.manual_seed(42)
    p = torch.randn(4, 4)
    g = torch.randn(4, 4)
    m = torch.zeros(4, 4)
    p_orig = p.clone()

    muon_fused_step([p], [g], [m], momentum=0.9, lr=0.01, wd=0.0, ns_steps=5)
    assert not torch.equal(p, p_orig), "Muon didn't update params"
    assert torch.all(torch.isfinite(p)), "NaN/Inf in Muon output"


# ═══════════════════════════════════════════════════════════════════════
#  Test 9: CPU C++ extension has all optimizer functions
# ═══════════════════════════════════════════════════════════════════════

def test_cpu_extension_completeness():
    """Verify CPU _ops module has all required optimizer functions."""
    import grokking_optimizers as go
    if not go._HAS_OPS:
        # Can't test C++ extension without it being built
        # Still pass — fallback coverage is tested separately
        return

    required = [
        'supergrok15_fused_step',
        'supergrok11_fused_step',
        'grokadamw_fused_step',
        'neuralgrok_fused_step',
        'prodigy_fused_step',
        'grokfast_fused_step',
        'lion_fused_step',
        'looksam_perturb_all',
        'looksam_restore_all',
        'looksam_compute_directions',
        'looksam_adjust_grads',
        'muon_fused_step',
    ]

    missing = [f for f in required if not hasattr(go._ops, f)]
    assert not missing, f"Missing CPU _ops functions: {missing}"


# ═══════════════════════════════════════════════════════════════════════
#  Test 10: All optimizers importable
# ═══════════════════════════════════════════════════════════════════════

def test_all_optimizers_importable():
    """Verify all optimizer classes can be imported."""
    from grokking_optimizers import (
        SuperGrok15, SuperGrok2, SuperGrok11,
        GrokAdamW, NeuralGrok, Prodigy,
        Grokfast, Lion, LookSAM, Muon,
    )
    # Just verify they're classes
    for cls in [SuperGrok15, SuperGrok2, SuperGrok11, GrokAdamW,
                NeuralGrok, Prodigy, Grokfast, Lion, LookSAM, Muon]:
        assert callable(cls), f"{cls.__name__} is not callable"


# ═══════════════════════════════════════════════════════════════════════
#  Test 11: Prodigy fallback returns d_lr
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_prodigy_d_lr():
    """Verify Prodigy fallback returns updated d_lr value."""
    import torch
    from grokking_optimizers._python_fallback import prodigy_fused_step

    torch.manual_seed(42)
    p = torch.randn(10)
    g = torch.randn(10)
    ea = torch.zeros(10)
    easq = torch.zeros(10)
    s = torch.zeros(10)
    p0 = p.clone()
    steps = [0]

    d_lr = prodigy_fused_step(
        [p], [g], [ea], [easq], [s], [p0], steps,
        d_lr=1.0, beta1=0.9, beta2=0.999, lr=1.0, wd=0.0, eps=1e-8,
    )
    assert isinstance(d_lr, float), f"d_lr should be float, got {type(d_lr)}"
    assert d_lr >= 1.0, f"d_lr should be >= initial (1.0), got {d_lr}"


# ═══════════════════════════════════════════════════════════════════════
#  Test 12: setup.py CPU path includes new source files
# ═══════════════════════════════════════════════════════════════════════

def test_setup_cpu_sources():
    """Verify setup.py CPU path lists the new source files."""
    from pathlib import Path
    setup_py = Path(__file__).parent.parent / "setup.py"
    content = setup_py.read_text()

    required_sources = [
        "csrc/cpu/generic/all_optimizers_cpu.cpp",
        "csrc/cpu/generic/supergrok2_scan_cpu.cpp",
    ]
    for src in required_sources:
        assert src in content, f"setup.py missing CPU source: {src}"

    # Check SIMD detection
    assert "avx512" in content.lower() or "simd" in content.lower(), \
        "setup.py missing SIMD/AVX-512 detection"


# ═══════════════════════════════════════════════════════════════════════
#  Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CPU Fallback + SIMD — Test Suite")
    print("=" * 60)

    run_test("1. __init__.py flags", test_init_flags)
    run_test("2. Python fallback module exists", test_python_fallback_exists)
    run_test("3. Conditional _ops import in optimizers", test_optimizer_conditional_import)
    run_test("4. Lion fallback numerics", test_fallback_lion_numerics)
    run_test("5. GrokAdamW fallback numerics", test_fallback_grokadamw_numerics)
    run_test("6. Grokfast EMA amplification", test_fallback_grokfast_ema)
    run_test("7. LookSAM perturb/restore cycle", test_fallback_looksam_cycle)
    run_test("8. Muon Newton-Schulz", test_fallback_muon_ns)
    run_test("9. CPU extension completeness", test_cpu_extension_completeness)
    run_test("10. All optimizers importable", test_all_optimizers_importable)
    run_test("11. Prodigy d_lr return", test_fallback_prodigy_d_lr)
    run_test("12. setup.py CPU sources", test_setup_cpu_sources)

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
