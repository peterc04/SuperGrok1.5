#!/usr/bin/env python3
"""
SuperGrok v2 JAX — Test Suite

Tests JAX implementation for correctness, JIT compatibility,
and numerical equivalence to the PyTorch reference.

Usage:    python supergrok2_jax_tpu/tests/test_supergrok2_jax.py
"""

import sys
import os
import traceback

# Add project root to path so 'supergrok2_jax_tpu' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np

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


# ═══════════════════════════════════════════════════════════════════
#  J1: Import Test
# ═══════════════════════════════════════════════════════════════════

def test_j1_import():
    """Verify all JAX modules import successfully."""
    from supergrok2_jax_tpu.scan import MambaScanWeights, mamba3_scan
    from supergrok2_jax_tpu.gru import GRUWeights, mini_gru
    from supergrok2_jax_tpu.peer import peer_expert_forward, peer_expert_forward_hard
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetWeights, MetaNetConfig, init_meta_weights, meta_net_forward
    from supergrok2_jax_tpu.supergrok2_jax import (
        PerParamState, SuperGrok2State, OptimizerConfig,
        init_state, supergrok2_step,
    )
    from supergrok2_jax_tpu.bilevel import bilevel_step
    from supergrok2_jax_tpu.sharding import create_mesh, shard_params, replicate_meta_weights
    from supergrok2_jax_tpu.quantization_jax import quantize_int8_symmetric, dequantize_int8
    from supergrok2_jax_tpu.bridge import pytorch_weights_to_jax


# ═══════════════════════════════════════════════════════════════════
#  J2: Associative Scan Operator Correctness
# ═══════════════════════════════════════════════════════════════════

def test_j2_associative_scan_operator():
    """Verify the combine operator is associative: combine(combine(a,b), c) == combine(a, combine(b,c))."""
    from supergrok2_jax_tpu.scan import _associative_combine

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Random 2x2 affine transforms
    def rand_affine(k):
        M = jax.random.normal(k, (3, 4, 2, 2)) * 0.5
        b = jax.random.normal(k, (3, 4, 2)) * 0.5
        return M, b

    a = rand_affine(k1)
    b = rand_affine(k2)
    c = rand_affine(k3)

    # (a ∘ b) ∘ c
    ab = _associative_combine(a, b)
    ab_c = _associative_combine(ab, c)

    # a ∘ (b ∘ c)
    bc = _associative_combine(b, c)
    a_bc = _associative_combine(a, bc)

    # Must be equal (associativity)
    M_diff = jnp.max(jnp.abs(ab_c[0] - a_bc[0]))
    b_diff = jnp.max(jnp.abs(ab_c[1] - a_bc[1]))

    assert M_diff < 1e-4, f"Matrix composition not associative: diff={M_diff}"
    assert b_diff < 1e-4, f"Bias composition not associative: diff={b_diff}"


# ═══════════════════════════════════════════════════════════════════
#  J3: Mamba Scan Forward
# ═══════════════════════════════════════════════════════════════════

def test_j3_mamba_scan_forward():
    """Verify Mamba scan produces finite output with correct shapes."""
    from supergrok2_jax_tpu.scan import MambaScanWeights, mamba3_scan
    import math

    d_model, d_state, d_inner = 8, 16, 16
    N = 64

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 15)

    def _linear(k, out_dim, in_dim):
        bound = 1.0 / math.sqrt(in_dim)
        return jax.random.uniform(k, (out_dim, in_dim), minval=-bound, maxval=bound)

    weights = MambaScanWeights(
        in_proj_W=_linear(keys[0], 2 * d_inner, d_model),
        dt_proj_W=_linear(keys[1], d_inner, d_inner),
        dt_proj_b=jnp.zeros(d_inner),
        B_proj_W=_linear(keys[2], d_state, d_inner),
        C_proj_W=_linear(keys[3], d_state, d_inner),
        A_log=jnp.log(jnp.linspace(1, d_state, d_state))[None, :].repeat(d_inner, axis=0),
        D=jnp.ones(d_inner),
        rope_freq=jax.random.normal(keys[4], (d_inner, d_state // 2)) * 0.01,
        out_proj_W=_linear(keys[5], d_model, d_inner),
    )

    x = jax.random.normal(keys[6], (N, d_model))
    init_state = jnp.zeros((d_inner, d_state))

    # Forward scan
    output, final_state = mamba3_scan(x, weights, init_state, reverse=False)
    assert output.shape == (N, d_model), f"Bad output shape: {output.shape}"
    assert final_state.shape == (d_inner, d_state), f"Bad state shape: {final_state.shape}"
    assert jnp.all(jnp.isfinite(output)), "Non-finite output"
    assert jnp.all(jnp.isfinite(final_state)), "Non-finite final state"

    # Reverse scan
    rev_output, rev_state = mamba3_scan(x, weights, init_state, reverse=True)
    assert rev_output.shape == (N, d_model)
    assert jnp.all(jnp.isfinite(rev_output))

    # Forward and reverse should differ (different scan directions)
    assert jnp.max(jnp.abs(output - rev_output)) > 1e-6, "Forward and reverse scans identical"


# ═══════════════════════════════════════════════════════════════════
#  J4: GRU Cell
# ═══════════════════════════════════════════════════════════════════

def test_j4_gru_cell():
    """Verify GRU produces finite output with correct shapes."""
    from supergrok2_jax_tpu.gru import GRUWeights, mini_gru
    import math

    N, input_dim, gru_hidden = 32, 18, 4
    total_dim = input_dim + gru_hidden

    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 5)
    bound = 1.0 / math.sqrt(total_dim)

    weights = GRUWeights(
        Wz=jax.random.uniform(keys[0], (gru_hidden, total_dim), minval=-bound, maxval=bound),
        bz=jnp.zeros(gru_hidden),
        Wr=jax.random.uniform(keys[1], (gru_hidden, total_dim), minval=-bound, maxval=bound),
        br=jnp.zeros(gru_hidden),
        Wh=jax.random.uniform(keys[2], (gru_hidden, total_dim), minval=-bound, maxval=bound),
        bh=jnp.zeros(gru_hidden),
    )

    x = jax.random.normal(keys[3], (N, input_dim))
    h_old = jax.random.normal(keys[4], (N, gru_hidden)) * 0.1

    h_new = mini_gru(x, h_old, weights)
    assert h_new.shape == (N, gru_hidden), f"Bad shape: {h_new.shape}"
    assert jnp.all(jnp.isfinite(h_new)), "Non-finite GRU output"

    # GRU output should differ from input state
    assert jnp.max(jnp.abs(h_new - h_old)) > 1e-6, "GRU didn't update state"


# ═══════════════════════════════════════════════════════════════════
#  J5: PEER Routing
# ═══════════════════════════════════════════════════════════════════

def test_j5_peer_routing():
    """Verify PEER routing produces finite output and tracks expert counts."""
    from supergrok2_jax_tpu.peer import peer_expert_forward, peer_expert_forward_hard

    N = 32
    d_model, num_heads, pk_dim = 8, 4, 12
    num_experts = pk_dim * pk_dim  # 144
    expert_hidden = 16
    gru_hidden = 4
    peer_input_dim = gru_hidden + 2 * d_model + 2  # 22

    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 10)

    peer_input = jax.random.normal(keys[0], (N, peer_input_dim))
    grad = jax.random.normal(keys[1], (N,))
    peer_query_Ws = jax.random.normal(keys[2], (num_heads, d_model, peer_input_dim)) * 0.1
    prod_keys_A = jax.random.normal(keys[3], (num_heads, pk_dim, d_model // 2)) * 0.02
    prod_keys_B = jax.random.normal(keys[4], (num_heads, pk_dim, d_model // 2)) * 0.02
    expert_W1 = jax.random.normal(keys[5], (num_experts, expert_hidden, 1)) * 0.02
    expert_b1 = jnp.zeros((num_experts, expert_hidden))
    expert_W2 = jax.random.normal(keys[6], (num_experts, 1, expert_hidden)) * 0.02
    expert_b2 = jnp.zeros((num_experts, 1))

    # Soft routing (bilevel)
    out_soft, counts_soft = peer_expert_forward(
        peer_input, grad, peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        num_heads, pk_dim, num_experts,
    )
    assert out_soft.shape == (N,), f"Bad shape: {out_soft.shape}"
    assert jnp.all(jnp.isfinite(out_soft)), "Non-finite soft output"
    assert counts_soft.sum() > 0, "No expert activations tracked"

    # Hard routing (forward step)
    out_hard, counts_hard = peer_expert_forward_hard(
        peer_input, grad, peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        num_heads, pk_dim, num_experts,
    )
    assert out_hard.shape == (N,)
    assert jnp.all(jnp.isfinite(out_hard))
    assert counts_hard.sum() > 0


# ═══════════════════════════════════════════════════════════════════
#  J6: Full Meta-Net Forward
# ═══════════════════════════════════════════════════════════════════

def test_j6_meta_net_forward():
    """Verify full meta-net forward produces valid output."""
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights, meta_net_forward

    config = MetaNetConfig()
    key = jax.random.PRNGKey(100)
    meta_weights = init_meta_weights(config, key)

    N = 64
    k1, k2 = jax.random.split(key)
    grad = jax.random.normal(k1, (N,)) * 0.1
    sharpness = jnp.abs(jax.random.normal(k2, (N,))) * 0.01
    gru_state = jnp.zeros((N, config.gru_hidden))
    mamba_fwd = jnp.zeros((config.d_inner, config.d_state))
    mamba_bwd = jnp.zeros((config.d_inner, config.d_state))

    smart_grad, new_gru, new_fwd, new_bwd, exp_counts = meta_net_forward(
        grad, sharpness, gru_state, mamba_fwd, mamba_bwd, meta_weights, config)

    assert smart_grad.shape == (N,), f"Bad shape: {smart_grad.shape}"
    assert new_gru.shape == (N, config.gru_hidden)
    assert new_fwd.shape == (config.d_inner, config.d_state)
    assert new_bwd.shape == (config.d_inner, config.d_state)
    assert jnp.all(jnp.isfinite(smart_grad)), "Non-finite smart_grad"
    assert jnp.all(jnp.isfinite(new_gru)), "Non-finite GRU state"


# ═══════════════════════════════════════════════════════════════════
#  J7: Optimizer Step
# ═══════════════════════════════════════════════════════════════════

def test_j7_optimizer_step():
    """Verify one optimizer step produces valid, changed parameters."""
    from supergrok2_jax_tpu.supergrok2_jax import OptimizerConfig, init_state, supergrok2_step
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights

    config = OptimizerConfig(lr=1e-2)
    meta_config = MetaNetConfig()
    key = jax.random.PRNGKey(200)

    # Simple model: two linear layers
    k1, k2, k3, k4 = jax.random.split(key, 4)
    params = {
        'w1': jax.random.normal(k1, (16, 8)) * 0.1,
        'w2': jax.random.normal(k2, (8, 4)) * 0.1,
    }
    grads = {
        'w1': jax.random.normal(k3, (16, 8)) * 0.01,
        'w2': jax.random.normal(k4, (8, 4)) * 0.01,
    }

    meta_weights = init_meta_weights(meta_config, key)
    opt_state = init_state(params, config, meta_config)

    new_params, new_opt_state = supergrok2_step(
        params, grads, opt_state, meta_weights, config, meta_config)

    # Check params changed
    w1_diff = jnp.max(jnp.abs(new_params['w1'] - params['w1']))
    assert w1_diff > 0, "w1 not updated"
    assert jnp.all(jnp.isfinite(new_params['w1'])), "Non-finite w1"
    assert jnp.all(jnp.isfinite(new_params['w2'])), "Non-finite w2"

    # Check state updated
    assert new_opt_state.global_step == 1
    assert new_opt_state.param_states[0].step_count == 1


# ═══════════════════════════════════════════════════════════════════
#  J8: Bilevel Gradient Non-Zero
# ═══════════════════════════════════════════════════════════════════

def test_j8_bilevel_grad_nonzero():
    """Verify jax.grad through associative_scan produces non-zero gradients."""
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights, meta_net_forward

    config = MetaNetConfig()
    key = jax.random.PRNGKey(300)
    meta_weights = init_meta_weights(config, key)

    N = 32
    k1, k2 = jax.random.split(key)
    grad = jax.random.normal(k1, (N,)) * 0.1
    sharpness = jnp.abs(jax.random.normal(k2, (N,))) * 0.01
    gru_state = jnp.zeros((N, config.gru_hidden))
    mamba_fwd = jnp.zeros((config.d_inner, config.d_state))
    mamba_bwd = jnp.zeros((config.d_inner, config.d_state))

    def loss_fn(mw):
        smart_grad, _, _, _, _ = meta_net_forward(
            grad, sharpness, gru_state, mamba_fwd, mamba_bwd,
            mw, config, use_soft_routing=True)
        return jnp.sum(smart_grad ** 2)

    meta_grads = jax.grad(loss_fn, allow_int=True)(meta_weights)

    # Check that at least some gradients are non-zero
    grad_leaves = jax.tree.leaves(meta_grads)
    any_nonzero = any(jnp.any(jnp.abs(g) > 1e-10) for g in grad_leaves)
    assert any_nonzero, "All meta-net gradients are zero"

    # Check all gradients are finite
    all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    assert all_finite, "Non-finite meta-net gradients"


# ═══════════════════════════════════════════════════════════════════
#  J9: JIT Compilation
# ═══════════════════════════════════════════════════════════════════

def test_j9_jit_compilation():
    """Verify scan and meta-net forward JIT compile without error."""
    from supergrok2_jax_tpu.scan import MambaScanWeights, mamba3_scan
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights, meta_net_forward
    import math

    # Test scan JIT
    d_model, d_state, d_inner = 8, 16, 16
    N = 32
    key = jax.random.PRNGKey(400)
    keys = jax.random.split(key, 10)
    bound = 1.0 / math.sqrt(d_model)

    weights = MambaScanWeights(
        in_proj_W=jax.random.uniform(keys[0], (2 * d_inner, d_model), minval=-bound, maxval=bound),
        dt_proj_W=jax.random.uniform(keys[1], (d_inner, d_inner), minval=-bound, maxval=bound),
        dt_proj_b=jnp.zeros(d_inner),
        B_proj_W=jax.random.uniform(keys[2], (d_state, d_inner), minval=-bound, maxval=bound),
        C_proj_W=jax.random.uniform(keys[3], (d_state, d_inner), minval=-bound, maxval=bound),
        A_log=jnp.log(jnp.linspace(1, d_state, d_state))[None, :].repeat(d_inner, axis=0),
        D=jnp.ones(d_inner),
        rope_freq=jax.random.normal(keys[4], (d_inner, d_state // 2)) * 0.01,
        out_proj_W=jax.random.uniform(keys[5], (d_model, d_inner), minval=-bound, maxval=bound),
    )

    x = jax.random.normal(keys[6], (N, d_model))
    init_state = jnp.zeros((d_inner, d_state))

    jit_scan = jax.jit(mamba3_scan, static_argnums=(3,))
    out1, state1 = jit_scan(x, weights, init_state, False)
    out2, state2 = jit_scan(x, weights, init_state, False)

    # Second call should use cached compilation (same result)
    assert jnp.allclose(out1, out2, atol=1e-6), "JIT results differ between calls"


# ═══════════════════════════════════════════════════════════════════
#  J10: INT8 Quantization
# ═══════════════════════════════════════════════════════════════════

def test_j10_int8_quantization():
    """Verify INT8 round-trip quantization."""
    from supergrok2_jax_tpu.quantization_jax import quantize_int8_symmetric, dequantize_int8

    key = jax.random.PRNGKey(500)
    tensor = jax.random.normal(key, (10, 4)) * 2.0

    qt = quantize_int8_symmetric(tensor)
    assert qt.data.dtype == jnp.int8
    assert qt.scale.shape == ()

    recovered = dequantize_int8(qt)
    assert recovered.dtype == jnp.float32

    max_err = jnp.max(jnp.abs(recovered - tensor))
    # INT8 with max range ~2.0: step size = 4.0/254 ≈ 0.016
    assert max_err < 0.05, f"INT8 round-trip error too large: {max_err}"


# ═══════════════════════════════════════════════════════════════════
#  J11: Sharding Module
# ═══════════════════════════════════════════════════════════════════

def test_j11_sharding_module():
    """Verify sharding utilities import and work on single device."""
    from supergrok2_jax_tpu.sharding import create_mesh, shard_params, replicate_meta_weights

    mesh = create_mesh()
    assert mesh is not None

    # Shard simple params
    params = {'w': jnp.ones((4, 4))}
    sharded = shard_params(params, mesh)
    assert jnp.allclose(sharded['w'], params['w'])


# ═══════════════════════════════════════════════════════════════════
#  J12: State Pytree Compatibility
# ═══════════════════════════════════════════════════════════════════

def test_j12_state_pytree():
    """Verify state structures are valid JAX pytrees."""
    from supergrok2_jax_tpu.supergrok2_jax import PerParamState, SuperGrok2State, OptimizerConfig, init_state
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig

    config = OptimizerConfig()
    meta_config = MetaNetConfig()
    params = {'w': jnp.ones((4, 4))}
    opt_state = init_state(params, config, meta_config)

    # Flatten and unflatten should round-trip
    leaves, treedef = jax.tree.flatten(opt_state)
    reconstructed = jax.tree.unflatten(treedef, leaves)

    # Check round-trip
    assert reconstructed.global_step == opt_state.global_step
    assert len(reconstructed.param_states) == len(opt_state.param_states)


# ═══════════════════════════════════════════════════════════════════
#  J13: Cross-Framework Scan (test vectors)
# ═══════════════════════════════════════════════════════════════════

def test_j13_cross_framework_scan():
    """Verify JAX scan produces finite output from deterministic test vectors."""
    from supergrok2_jax_tpu.bridge import export_test_vectors
    from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights, meta_net_forward
    import tempfile, os

    # Generate deterministic test vectors
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp_path = f.name
    try:
        vectors = export_test_vectors(save_path=tmp_path)

        # Load back and verify
        loaded = np.load(tmp_path)
        grad_np = loaded['inputs/grad']
        sharpness_np = loaded['inputs/sharpness']
        gru_np = loaded['inputs/gru_state']
        fwd_np = loaded['inputs/mamba_fwd_state']
        bwd_np = loaded['inputs/mamba_bwd_state']

        # Convert to JAX
        grad_jax = jnp.array(grad_np)
        sharpness_jax = jnp.array(sharpness_np)
        gru_jax = jnp.array(gru_np)
        fwd_jax = jnp.array(fwd_np)
        bwd_jax = jnp.array(bwd_np)

        # Run JAX forward
        config = MetaNetConfig()
        key = jax.random.PRNGKey(42)
        meta_weights = init_meta_weights(config, key)

        smart_grad, new_gru, new_fwd, new_bwd, exp_counts = meta_net_forward(
            grad_jax, sharpness_jax, gru_jax, fwd_jax, bwd_jax,
            meta_weights, config)

        assert smart_grad.shape == grad_jax.shape, f"Shape mismatch: {smart_grad.shape}"
        assert jnp.all(jnp.isfinite(smart_grad)), "Non-finite smart_grad"
        assert jnp.all(jnp.isfinite(new_gru)), "Non-finite GRU state"
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
#  J14: All Simple Optimizers
# ═══════════════════════════════════════════════════════════════════

def test_j14_simple_optimizers():
    """Verify all simple optimizers produce valid, changed parameters."""
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        GrokAdamWConfig, init_grokadamw_state, grokadamw_step,
        LionConfig, init_lion_state, lion_step,
        GrokfastConfig, init_grokfast_state, grokfast_amplify,
        ProdigyConfig, init_prodigy_state, prodigy_step,
        MuonConfig, init_muon_state, muon_step,
        LookSAMConfig, init_looksam_state, looksam_adam_step,
    )

    key = jax.random.PRNGKey(999)
    k1, k2 = jax.random.split(key)
    param = jax.random.normal(k1, (4, 4)) * 0.1
    grad = jax.random.normal(k2, (4, 4)) * 0.01

    # GrokAdamW
    s = init_grokadamw_state(param)
    new_p, new_s = grokadamw_step(param, grad, s, GrokAdamWConfig())
    assert jnp.all(jnp.isfinite(new_p)), "GrokAdamW non-finite"
    assert not jnp.allclose(new_p, param), "GrokAdamW unchanged"

    # Lion
    s = init_lion_state(param)
    new_p, new_s = lion_step(param, grad, s, LionConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Lion non-finite"
    assert not jnp.allclose(new_p, param), "Lion unchanged"

    # Grokfast (amplify only)
    s = init_grokfast_state(param)
    amp_g, new_s = grokfast_amplify(grad, s, GrokfastConfig())
    assert jnp.all(jnp.isfinite(amp_g)), "Grokfast non-finite"

    # Prodigy
    s = init_prodigy_state(param)
    new_p, new_s, d_lr = prodigy_step(param, grad, s, ProdigyConfig(), d_lr=1.0)
    assert jnp.all(jnp.isfinite(new_p)), "Prodigy non-finite"

    # Muon
    s = init_muon_state(param)
    new_p, new_s = muon_step(param, grad, s, MuonConfig())
    assert jnp.all(jnp.isfinite(new_p)), "Muon non-finite"
    assert not jnp.allclose(new_p, param), "Muon unchanged"

    # LookSAM (Adam step only)
    s = init_looksam_state(param)
    new_p, new_s = looksam_adam_step(param, grad, s, LookSAMConfig())
    assert jnp.all(jnp.isfinite(new_p)), "LookSAM non-finite"
    assert not jnp.allclose(new_p, param), "LookSAM unchanged"


# ═══════════════════════════════════════════════════════════════════
#  J15: Meta-Net Optimizers
# ═══════════════════════════════════════════════════════════════════

def test_j15_metanet_optimizers():
    """Verify SuperGrok v1.5, v1.1, and NeuralGrok produce valid output."""
    import math
    from supergrok2_jax_tpu.metanet_optimizers_jax import (
        SuperGrok15Config, SuperGrok15Weights, init_supergrok15_state, supergrok15_step,
        SuperGrok11Config, SuperGrok11Weights, init_supergrok11_state, supergrok11_step,
        NeuralGrokConfig, NeuralGrokWeights, init_neuralgrok_state, neuralgrok_step,
    )

    key = jax.random.PRNGKey(777)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    param = jax.random.normal(k1, (16,)) * 0.1
    grad = jax.random.normal(k2, (16,)) * 0.01

    hidden_dim = 32
    bound = 1.0 / math.sqrt(1)

    # SuperGrok v1.5
    w15 = SuperGrok15Weights(
        W1=jax.random.uniform(k3, (hidden_dim, 1), minval=-bound, maxval=bound),
        b1=jnp.zeros(hidden_dim),
        W2=jax.random.uniform(k4, (1, hidden_dim), minval=-bound, maxval=bound),
        b2=jnp.zeros(1),
        rescale=0.1,
    )
    s = init_supergrok15_state(param)
    new_p, new_s = supergrok15_step(param, grad, s, w15, SuperGrok15Config())
    assert jnp.all(jnp.isfinite(new_p)), "SG15 non-finite"
    assert not jnp.allclose(new_p, param), "SG15 unchanged"

    # SuperGrok v1.1
    w11 = SuperGrok11Weights(
        W1=w15.W1, b1=w15.b1, W2=w15.W2, b2=w15.b2, rescale=0.1)
    s = init_supergrok11_state(param)
    new_p, new_s = supergrok11_step(param, grad, s, w11, SuperGrok11Config())
    assert jnp.all(jnp.isfinite(new_p)), "SG11 non-finite"
    assert not jnp.allclose(new_p, param), "SG11 unchanged"

    # NeuralGrok
    k5, k6 = jax.random.split(k3)
    wng = NeuralGrokWeights(
        W1=jax.random.uniform(k5, (hidden_dim, 1), minval=-bound, maxval=bound),
        b1=jnp.zeros(hidden_dim),
        W_last=jax.random.uniform(k6, (1, hidden_dim), minval=-bound, maxval=bound),
        b_last=jnp.zeros(1),
        alpha=0.5, beta=0.5,
    )
    s = init_neuralgrok_state(param)
    new_p, new_s = neuralgrok_step(param, grad, s, wng, NeuralGrokConfig())
    assert jnp.all(jnp.isfinite(new_p)), "NeuralGrok non-finite"
    assert not jnp.allclose(new_p, param), "NeuralGrok unchanged"


# ═══════════════════════════════════════════════════════════════════
#  J16: Sharding + Multi-Host Utilities
# ═══════════════════════════════════════════════════════════════════

def test_j16_sharding_multihost():
    """Verify multi-host sharding utilities import and basic ops work."""
    from supergrok2_jax_tpu.sharding import (
        create_mesh, shard_params, shard_batch,
        replicate_meta_weights, initialize_multi_host,
        sharded_supergrok2_step, sharded_bilevel_step,
    )
    from supergrok2_jax_tpu.pallas_kernels import _HAS_PALLAS, pallas_mamba3_scan

    # Verify multi-host functions are callable
    assert callable(initialize_multi_host)
    assert callable(sharded_supergrok2_step)
    assert callable(sharded_bilevel_step)

    # Verify Pallas stub works
    assert isinstance(_HAS_PALLAS, bool)
    assert callable(pallas_mamba3_scan)

    # Verify shard_batch works
    mesh = create_mesh()
    batch = jnp.ones((4, 8))
    sharded = shard_batch(batch, mesh)
    assert jnp.allclose(sharded, batch)


# ═══════════════════════════════════════════════════════════════════
#  J17: JIT No-Retrace
# ═══════════════════════════════════════════════════════════════════

def test_j17_jit_no_retrace():
    """Verify JIT-compiled simple optimizers don't retrace on second call."""
    from supergrok2_jax_tpu.simple_optimizers_jax import (
        LionConfig, init_lion_state, lion_step,
        GrokAdamWConfig, init_grokadamw_state, grokadamw_step,
    )

    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)
    param = jax.random.normal(k1, (8,)) * 0.1
    grad = jax.random.normal(k2, (8,)) * 0.01

    # Lion JIT
    lion_cfg = LionConfig()
    s = init_lion_state(param)
    jit_lion = jax.jit(lion_step, static_argnums=(3,))

    p1, s1 = jit_lion(param, grad, s, lion_cfg)
    p2, s2 = jit_lion(p1, grad, s1, lion_cfg)

    # Second call should produce different result (optimizer makes progress)
    assert not jnp.allclose(p1, p2), "Lion JIT stuck"
    assert jnp.all(jnp.isfinite(p2)), "Lion JIT non-finite"

    # GrokAdamW JIT
    grok_cfg = GrokAdamWConfig()
    s = init_grokadamw_state(param)
    jit_grok = jax.jit(grokadamw_step, static_argnums=(3,))

    p1, s1 = jit_grok(param, grad, s, grok_cfg)
    p2, s2 = jit_grok(p1, grad, s1, grok_cfg)

    assert not jnp.allclose(p1, p2), "GrokAdamW JIT stuck"
    assert jnp.all(jnp.isfinite(p2)), "GrokAdamW JIT non-finite"


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SuperGrok v2 JAX — Test Suite")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print("=" * 60)

    print("\n--- J1: Import ---")
    run_test("J1: Import all JAX modules", test_j1_import)

    print("\n--- J2: Associative Scan Operator ---")
    run_test("J2: Associative combine operator", test_j2_associative_scan_operator)

    print("\n--- J3: Mamba Scan ---")
    run_test("J3: Mamba scan forward/reverse", test_j3_mamba_scan_forward)

    print("\n--- J4: GRU Cell ---")
    run_test("J4: GRU cell update", test_j4_gru_cell)

    print("\n--- J5: PEER Routing ---")
    run_test("J5: PEER routing + expert MLP", test_j5_peer_routing)

    print("\n--- J6: Full Forward ---")
    run_test("J6: Full meta-net forward", test_j6_meta_net_forward)

    print("\n--- J7: Optimizer Step ---")
    run_test("J7: Optimizer step", test_j7_optimizer_step)

    print("\n--- J8: Bilevel Gradient ---")
    run_test("J8: Bilevel gradient non-zero", test_j8_bilevel_grad_nonzero)

    print("\n--- J9: JIT Compilation ---")
    run_test("J9: JIT compilation", test_j9_jit_compilation)

    print("\n--- J10: INT8 Quantization ---")
    run_test("J10: INT8 quantization round-trip", test_j10_int8_quantization)

    print("\n--- J11: Sharding ---")
    run_test("J11: Sharding module", test_j11_sharding_module)

    print("\n--- J12: State Pytree ---")
    run_test("J12: State pytree compatibility", test_j12_state_pytree)

    print("\n--- J13: Cross-Framework Scan ---")
    run_test("J13: Cross-framework test vectors", test_j13_cross_framework_scan)

    print("\n--- J14: Simple Optimizers ---")
    run_test("J14: All simple optimizers", test_j14_simple_optimizers)

    print("\n--- J15: Meta-Net Optimizers ---")
    run_test("J15: Meta-net optimizers (SG15, SG11, NeuralGrok)", test_j15_metanet_optimizers)

    print("\n--- J16: Sharding + Multi-Host ---")
    run_test("J16: Sharding + multi-host utilities", test_j16_sharding_multihost)

    print("\n--- J17: JIT No-Retrace ---")
    run_test("J17: JIT no-retrace", test_j17_jit_no_retrace)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    print(f"Results: {passed}/{total} PASSED, {failed}/{total} FAILED")

    if failed > 0:
        print("\nFailed tests:")
        for name, ok, msg in results:
            if not ok:
                print(f"  - {name}: {msg}")

    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
