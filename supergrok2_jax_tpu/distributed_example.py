#!/usr/bin/env python3
"""
Multi-TPU Pod Training Example for SuperGrok v2 (JAX)

Demonstrates data-parallel training across a TPU pod (multiple hosts).
Each host runs this script; JAX distributed runtime handles communication.

Usage (single host, for testing):
    python distributed_example.py

Usage (TPU v4-32 pod, 4 hosts):
    # Run on each host — JAX auto-detects TPU pod topology
    python distributed_example.py

Requirements:
    - jax[tpu] installed
    - On TPU pods: environment variables set by TPU VM runtime
"""

import jax
import jax.numpy as jnp
from functools import partial

from supergrok2_jax_tpu import (
    OptimizerConfig, init_state, supergrok2_step,
    init_meta_weights, MetaNetWeights,
)
from supergrok2_jax_tpu.mamba3_peer_metanet_jax import MetaNetConfig
from supergrok2_jax_tpu.sharding import (
    initialize_multi_host,
    create_mesh,
    shard_params,
    shard_batch,
    replicate_meta_weights,
)


def simple_model(params, x):
    """Two-layer MLP for demonstration."""
    h = x @ params['w1']
    h = jax.nn.relu(h)
    return h @ params['w2']


def cross_entropy_loss(logits, labels):
    """Simple cross-entropy loss."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))


def main():
    # ── 1. Initialize distributed runtime ────────────────────────────
    initialize_multi_host()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Local devices: {jax.local_devices()}")
    print(f"Total devices: {jax.device_count()}")
    print(f"Process index: {jax.process_index()}/{jax.process_count()}")

    # ── 2. Create mesh ───────────────────────────────────────────────
    mesh = create_mesh(axis_name='dp')
    print(f"Mesh: {mesh}")

    # ── 3. Initialize model + optimizer ──────────────────────────────
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Simple 2-layer model
    params = {
        'w1': jax.random.normal(k1, (16, 32)) * 0.1,
        'w2': jax.random.normal(k2, (32, 4)) * 0.1,
    }

    config = OptimizerConfig(lr=1e-3)
    meta_config = MetaNetConfig()
    meta_weights = init_meta_weights(meta_config, k3)
    opt_state = init_state(params, config, meta_config)

    # ── 4. Shard data across devices ─────────────────────────────────
    params = shard_params(params, mesh)
    meta_weights = replicate_meta_weights(meta_weights, mesh)

    # ── 5. Define training step ──────────────────────────────────────
    @jax.jit
    def train_step(params, opt_state, meta_weights, x, y):
        def loss_fn(p):
            logits = simple_model(p, x)
            return cross_entropy_loss(logits, y)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        new_params, new_opt_state = supergrok2_step(
            params, grads, opt_state, meta_weights, config, meta_config)
        return new_params, new_opt_state, loss_val

    # ── 6. Training loop ─────────────────────────────────────────────
    n_steps = 5
    batch_size = 8 * jax.device_count()  # scale with device count

    for step in range(n_steps):
        # Synthetic data
        data_key = jax.random.PRNGKey(step)
        dk1, dk2 = jax.random.split(data_key)
        x = jax.random.normal(dk1, (batch_size, 16))
        y = jax.nn.one_hot(
            jax.random.randint(dk2, (batch_size,), 0, 4), 4)

        # Shard batch across devices
        x = shard_batch(x, mesh)
        y = shard_batch(y, mesh)

        params, opt_state, loss = train_step(
            params, opt_state, meta_weights, x, y)

        if jax.process_index() == 0:
            print(f"Step {step}: loss = {float(loss):.4f}")

    if jax.process_index() == 0:
        print("Training complete!")


if __name__ == '__main__':
    main()
