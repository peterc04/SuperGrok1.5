#!/usr/bin/env python3
"""JAX optimizer test matrix.

Tests all JAX optimizers with basic sanity checks.
Separate from test_matrix.py because JAX and PyTorch don't mix well in one process.

Usage:
    python tests/test_jax_matrix.py
"""

import sys
import traceback

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

results = []


def run_test(name, fn):
    """Run a test function, record PASS/FAIL."""
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, str(e)))
        print(f"  FAIL  {name} -- {e}")
        print(tb)


def test_grokadamw():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        GrokAdamWConfig, init_grokadamw_state, grokadamw_step,
    )
    key = jax.random.PRNGKey(1)
    p = jax.random.normal(key, (64,)) * 0.1
    g = jax.random.normal(key, (64,)) * 0.01
    s = init_grokadamw_state(p)
    new_p, new_s = grokadamw_step(p, g, s, GrokAdamWConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"
    assert new_s.step == 1, "Step not incremented"


def test_lion():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        LionConfig, init_lion_state, lion_step,
    )
    key = jax.random.PRNGKey(2)
    p = jax.random.normal(key, (64,)) * 0.1
    g = jax.random.normal(key, (64,)) * 0.01
    s = init_lion_state(p)
    new_p, new_s = lion_step(p, g, s, LionConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_grokfast():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        GrokfastConfig, init_grokfast_state, grokfast_amplify,
    )
    key = jax.random.PRNGKey(3)
    g = jax.random.normal(key, (64,)) * 0.01
    s = init_grokfast_state(g)
    amp_g, new_s = grokfast_amplify(g, s, GrokfastConfig())
    assert jnp.all(jnp.isfinite(amp_g)), "Non-finite"
    # Amplified gradient should differ from original
    assert not jnp.allclose(amp_g, g), "Not amplified"


def test_prodigy():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        ProdigyConfig, init_prodigy_state, prodigy_step,
    )
    key = jax.random.PRNGKey(4)
    p = jax.random.normal(key, (64,)) * 0.1
    g = jax.random.normal(key, (64,)) * 0.01
    s = init_prodigy_state(p)
    new_p, new_s, d_lr = prodigy_step(p, g, s, ProdigyConfig(), d_lr=1.0)
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert isinstance(d_lr, float), "d_lr should be float"


def test_muon():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        MuonConfig, init_muon_state, muon_step,
    )
    key = jax.random.PRNGKey(5)
    # Muon needs 2D params for Newton-Schulz
    p = jax.random.normal(key, (8, 8)) * 0.1
    g = jax.random.normal(key, (8, 8)) * 0.01
    s = init_muon_state(p)
    new_p, new_s = muon_step(p, g, s, MuonConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_looksam():
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        LookSAMConfig, init_looksam_state, looksam_adam_step,
        looksam_perturb, looksam_compute_direction,
    )
    key = jax.random.PRNGKey(6)
    p = jax.random.normal(key, (64,)) * 0.1
    g = jax.random.normal(key, (64,)) * 0.01
    s = init_looksam_state(p)

    # Test perturb
    p_perturbed = looksam_perturb(p, g, rho=0.05)
    assert not jnp.allclose(p_perturbed, p), "Perturb didn't change params"

    # Test direction
    g2 = g + jax.random.normal(key, (64,)) * 0.001
    d = looksam_compute_direction(g2, g)
    assert jnp.all(jnp.isfinite(d)), "Non-finite direction"

    # Test Adam step
    new_p, new_s = looksam_adam_step(p, g, s, LookSAMConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_supergrok15():
    import math
    from supergrok2_jax_tpu.metanet_optimizers_jax import (
        SuperGrok15Config, SuperGrok15Weights,
        init_supergrok15_state, supergrok15_step,
    )
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    p = jax.random.normal(k1, (32,)) * 0.1
    g = jax.random.normal(k2, (32,)) * 0.01
    w = SuperGrok15Weights(
        W1=jax.random.normal(k1, (32, 1)) * 0.1,
        b1=jnp.zeros(32),
        W2=jax.random.normal(k2, (1, 32)) * 0.1,
        b2=jnp.zeros(1),
        rescale=0.1,
    )
    s = init_supergrok15_state(p)
    new_p, new_s = supergrok15_step(p, g, s, w, SuperGrok15Config())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_supergrok11():
    from supergrok2_jax_tpu.metanet_optimizers_jax import (
        SuperGrok11Config, SuperGrok11Weights,
        init_supergrok11_state, supergrok11_step,
    )
    key = jax.random.PRNGKey(8)
    k1, k2 = jax.random.split(key)
    p = jax.random.normal(k1, (32,)) * 0.1
    g = jax.random.normal(k2, (32,)) * 0.01
    w = SuperGrok11Weights(
        W1=jax.random.normal(k1, (32, 1)) * 0.1,
        b1=jnp.zeros(32),
        W2=jax.random.normal(k2, (1, 32)) * 0.1,
        b2=jnp.zeros(1),
        rescale=0.1,
    )
    s = init_supergrok11_state(p)
    new_p, new_s = supergrok11_step(p, g, s, w, SuperGrok11Config())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_neuralgrok():
    from supergrok2_jax_tpu.metanet_optimizers_jax import (
        NeuralGrokConfig, NeuralGrokWeights,
        init_neuralgrok_state, neuralgrok_step,
    )
    key = jax.random.PRNGKey(9)
    k1, k2 = jax.random.split(key)
    p = jax.random.normal(k1, (32,)) * 0.1
    g = jax.random.normal(k2, (32,)) * 0.01
    w = NeuralGrokWeights(
        W1=jax.random.normal(k1, (32, 1)) * 0.1,
        b1=jnp.zeros(32),
        W_last=jax.random.normal(k2, (1, 32)) * 0.1,
        b_last=jnp.zeros(1),
        alpha=0.5, beta=0.5,
    )
    s = init_neuralgrok_state(p)
    new_p, new_s = neuralgrok_step(p, g, s, w, NeuralGrokConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Non-finite"
    assert not jnp.allclose(new_p, p), "Unchanged"


def test_supergrok2():
    """Lightweight SG2 check — full tests in test_supergrok2_jax.py."""
    from supergrok2_jax_tpu import (
        OptimizerConfig, init_state, supergrok2_step,
    )
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights

    key = jax.random.PRNGKey(10)
    k1, k2 = jax.random.split(key)
    params = {'w': jax.random.normal(k1, (16, 8)) * 0.1}
    grads = {'w': jax.random.normal(k2, (16, 8)) * 0.01}

    config = OptimizerConfig(lr=1e-2)
    meta_config = MetaNetConfig()
    meta_weights = init_meta_weights(meta_config, key)
    opt_state = init_state(params, config, meta_config)

    new_params, new_state = supergrok2_step(
        params, grads, opt_state, meta_weights, config, meta_config)
    assert jnp.all(jnp.isfinite(new_params['w'])), "Non-finite"
    assert not jnp.allclose(new_params['w'], params['w']), "Unchanged"


def main():
    if not _HAS_JAX:
        print("JAX not installed -- skipping JAX matrix tests")
        return True

    print(f"\n{'='*60}")
    print(f"  JAX Optimizer Test Matrix")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(f"{'='*60}\n")

    run_test("GrokAdamW (JAX)", test_grokadamw)
    run_test("Lion (JAX)", test_lion)
    run_test("Grokfast (JAX)", test_grokfast)
    run_test("Prodigy (JAX)", test_prodigy)
    run_test("Muon (JAX)", test_muon)
    run_test("LookSAM (JAX)", test_looksam)
    run_test("SuperGrok v1.5 (JAX)", test_supergrok15)
    run_test("SuperGrok v1.1 (JAX)", test_supergrok11)
    run_test("NeuralGrok (JAX)", test_neuralgrok)
    run_test("SuperGrok v2 (JAX)", test_supergrok2)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    print(f"\n{'─'*60}")
    print(f"  Results: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print(f"{'─'*60}\n")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
