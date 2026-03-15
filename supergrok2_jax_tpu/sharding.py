"""
TPU Sharding Annotations for SuperGrok v2 (JAX)

Provides utilities for sharding the optimizer across TPU chips.
Model parameters and optimizer states are sharded along the data-parallel axis.
Meta-net weights are replicated (small, needed on every chip).

Supports both single-host and multi-host TPU pod configurations.

Usage::

    # Single-host
    mesh = create_mesh()
    sharded_params = shard_params(params, mesh)
    replicated_meta = replicate_meta_weights(meta_weights, mesh)

    # Multi-host TPU pod
    initialize_multi_host()  # call once at startup
    mesh = create_mesh()
    sharded_params = shard_params(params, mesh)
"""

import jax
import jax.numpy as jnp
from typing import Any, Optional, Callable


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


# ═════════════════════════════════════════════════════════════════════════
#  Multi-Host TPU Pod Support (Prompt D §4)
# ═════════════════════════════════════════════════════════════════════════

def initialize_multi_host(
    coordinator_address: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
) -> None:
    """Initialize JAX distributed runtime for multi-host TPU pods.

    Call once at program startup before any JAX operations.
    On single-host setups, this is a no-op.

    Environment variables (auto-detected on TPU VMs):
        - TPU_CHIPS_PER_HOST_BOUNDS
        - TPU_HOST_BOUNDS
        - CLOUD_TPU_TASK_ID
        - TPU_WORKER_HOSTNAMES

    Args:
        coordinator_address: IP:port of coordinator (auto-detected on TPU VMs)
        num_processes: total number of hosts (auto-detected on TPU VMs)
        process_id: this host's ID (auto-detected on TPU VMs)
    """
    import os
    # Auto-detect if we're on a TPU pod (multi-host)
    is_tpu_pod = os.environ.get('TPU_WORKER_HOSTNAMES', '') != ''
    is_multi_process = (num_processes is not None and num_processes > 1)

    if is_tpu_pod or is_multi_process:
        kwargs = {}
        if coordinator_address is not None:
            kwargs['coordinator_address'] = coordinator_address
        if num_processes is not None:
            kwargs['num_processes'] = num_processes
        if process_id is not None:
            kwargs['process_id'] = process_id
        jax.distributed.initialize(**kwargs)


def sharded_supergrok2_step(
    params: Any,
    grads: Any,
    opt_state: Any,
    meta_weights: Any,
    config: Any,
    meta_config: Any,
    step_fn: Callable,
    axis_name: str = 'dp',
) -> Any:
    """SuperGrok v2 step with multi-device gradient aggregation.

    Wraps the base step function with gradient pmean for data parallelism.
    Use inside jax.pmap or shard_map with the matching axis_name.

    Args:
        params: model parameters (replicated)
        grads: per-device gradients (sharded)
        opt_state: optimizer state
        meta_weights: meta-net weights (replicated)
        config: OptimizerConfig
        meta_config: MetaNetConfig
        step_fn: the base supergrok2_step function
        axis_name: name of the DP axis for pmean

    Returns:
        (new_params, new_opt_state) with synchronized gradients
    """
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name=axis_name)
    return step_fn(params, grads, opt_state, meta_weights, config, meta_config)


def sharded_bilevel_step(
    params: Any,
    train_grads: Any,
    val_x: jnp.ndarray,
    val_y: jnp.ndarray,
    opt_state: Any,
    meta_weights: Any,
    config: Any,
    meta_config: Any,
    model_fn: Callable,
    loss_fn: Callable,
    bilevel_fn: Callable,
    meta_lr: float = 1e-4,
    axis_name: str = 'dp',
) -> Any:
    """Bilevel step with multi-device gradient aggregation.

    Wraps the base bilevel_step with gradient pmean for data parallelism.

    Args:
        params: model parameters (replicated)
        train_grads: per-device training gradients
        val_x: validation inputs (sharded along batch dim)
        val_y: validation targets (sharded along batch dim)
        opt_state: optimizer state
        meta_weights: meta-net weights (replicated)
        config: OptimizerConfig
        meta_config: MetaNetConfig
        model_fn: model forward function
        loss_fn: loss function
        bilevel_fn: the base bilevel_step function
        meta_lr: meta-learning rate
        axis_name: name of the DP axis

    Returns:
        (new_meta_weights, meta_loss_val)
    """
    # Average training gradients across devices
    train_grads = jax.lax.pmean(train_grads, axis_name=axis_name)

    new_meta_weights, meta_loss = bilevel_fn(
        params, train_grads, val_x, val_y,
        opt_state, meta_weights, config, meta_config,
        model_fn, loss_fn, meta_lr,
    )

    # Average meta-loss for logging
    meta_loss = jax.lax.pmean(meta_loss, axis_name=axis_name)

    return new_meta_weights, meta_loss
