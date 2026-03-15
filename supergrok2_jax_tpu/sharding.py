"""
TPU Sharding Annotations for SuperGrok v2 (JAX)

Provides utilities for sharding the optimizer across TPU chips.
Model parameters and optimizer states are sharded along the data-parallel axis.
Meta-net weights are replicated (small, needed on every chip).

Usage::

    mesh = create_mesh()
    sharded_params = shard_params(params, mesh)
    replicated_meta = replicate_meta_weights(meta_weights, mesh)

    @jax.jit
    def train_step(params, grads, opt_state, meta_weights):
        return supergrok2_step(params, grads, opt_state, meta_weights, config)

    params, opt_state = train_step(sharded_params, grads, opt_state, replicated_meta)
"""

import jax
import jax.numpy as jnp
from typing import Any, Optional


def create_mesh(
    axis_name: str = 'dp',
    devices: Optional[Any] = None,
) -> jax.sharding.Mesh:
    """Create a JAX mesh for data-parallel training.

    Args:
        axis_name: name of the data-parallel axis
        devices: optional list of devices (default: all available)

    Returns:
        jax.sharding.Mesh
    """
    if devices is None:
        devices = jax.devices()

    return jax.sharding.Mesh(devices, (axis_name,))


def shard_params(
    params: Any,
    mesh: jax.sharding.Mesh,
    axis_name: str = 'dp',
) -> Any:
    """Shard model parameters across the data-parallel axis.

    Each device gets a full copy of the parameters (for data parallelism,
    parameters are replicated, not sharded).

    Args:
        params: pytree of model parameters
        mesh: JAX mesh
        axis_name: name of the DP axis

    Returns:
        sharded params on the mesh
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    replicated = NamedSharding(mesh, P())
    return jax.device_put(params, replicated)


def replicate_meta_weights(
    meta_weights: Any,
    mesh: jax.sharding.Mesh,
) -> Any:
    """Replicate meta-net weights across all devices.

    Meta-net weights are small (~100KB) and needed on every chip.
    They should NOT be sharded.

    Args:
        meta_weights: MetaNetWeights
        mesh: JAX mesh

    Returns:
        replicated meta_weights
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    replicated = NamedSharding(mesh, P())
    return jax.device_put(meta_weights, replicated)


def shard_batch(
    batch: Any,
    mesh: jax.sharding.Mesh,
    axis_name: str = 'dp',
) -> Any:
    """Shard a data batch across the data-parallel axis.

    The leading dimension (batch dim) is split across devices.

    Args:
        batch: pytree with arrays that have a batch dimension
        mesh: JAX mesh
        axis_name: name of the DP axis

    Returns:
        sharded batch
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharded = NamedSharding(mesh, P(axis_name))
    return jax.device_put(batch, sharded)


def all_gather_grad_for_scan(
    grad_shard: jnp.ndarray,
    axis_name: str = 'dp',
) -> jnp.ndarray:
    """All-gather gradient across data-parallel axis for meta-net scan.

    The Mamba scan needs all N elements. When parameters are sharded
    across devices, each device only has N/num_devices elements.
    This function gathers the full gradient before scanning.

    Must be called inside jax.pmap or shard_map with the matching axis_name.

    Args:
        grad_shard: [N_local] local gradient shard
        axis_name: name of the DP axis

    Returns:
        full_grad: [N_total] concatenated gradient
    """
    return jax.lax.all_gather(grad_shard, axis_name=axis_name, axis=0)
