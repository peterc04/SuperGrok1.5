"""
SuperGrok v2 — JAX/TPU Implementation

A clean JAX rewrite of the SuperGrok v2 optimizer for TPU (and JAX-on-GPU).
Uses JAX's native primitives (lax.associative_scan, jax.grad, jax.jit) instead
of hand-written CUDA kernels.

Key differences from the PyTorch/CUDA version:
  - Functional: no in-place mutation; all state is explicit (passed in, returned out)
  - Parallel scan: lax.associative_scan replaces 500 lines of Blelloch CUDA
  - Automatic differentiation: jax.grad replaces manual backward kernels for bilevel
  - TPU native: XLA compiler handles memory layout and hardware mapping

Usage::

    import jax
    import jax.numpy as jnp
    from supergrok2_jax_tpu import (
        SuperGrok2State, MetaNetWeights, OptimizerConfig,
        init_state, supergrok2_step,
    )

    # Initialize
    config = OptimizerConfig()
    meta_weights = init_meta_weights(config, key=jax.random.PRNGKey(0))
    opt_state = init_state(params, config)

    # Training loop
    for step in range(n_steps):
        grads = jax.grad(loss_fn)(params)
        params, opt_state = supergrok2_step(params, grads, opt_state, meta_weights, config)
"""

__version__ = "2.1.0"

from .supergrok2_jax import (
    PerParamState,
    SuperGrok2State,
    OptimizerConfig,
    init_state,
    init_per_param_state,
    supergrok2_step,
)
from .mamba3_peer_metanet_jax import (
    MetaNetWeights,
    MambaScanWeights,
    GRUWeights,
    init_meta_weights,
    meta_net_forward,
)
from .scan import mamba3_scan
from .gru import mini_gru
from .peer import peer_expert_forward
from .bilevel import bilevel_step
from .sharding import create_mesh, shard_params, replicate_meta_weights
from .bridge import pytorch_weights_to_jax, jax_weights_to_pytorch

# Simple optimizers (Prompt D §1)
from .simple_optimizers_jax import (
    GrokAdamWState, GrokAdamWConfig, init_grokadamw_state, grokadamw_step,
    LionState, LionConfig, init_lion_state, lion_step,
    GrokfastState, GrokfastConfig, init_grokfast_state, grokfast_amplify,
    ProdigyState, ProdigyConfig, init_prodigy_state, prodigy_step,
    MuonState, MuonConfig, init_muon_state, muon_step,
    LookSAMState, LookSAMConfig, init_looksam_state,
    looksam_perturb, looksam_compute_direction, looksam_adjust_grad,
    looksam_adam_step,
)

# Meta-net optimizers (Prompt D §2)
from .metanet_optimizers_jax import (
    SuperGrok15State, SuperGrok15Weights, SuperGrok15Config,
    init_supergrok15_state, supergrok15_step,
    SuperGrok11State, SuperGrok11Weights, SuperGrok11Config,
    init_supergrok11_state, supergrok11_step,
    NeuralGrokState, NeuralGrokWeights, NeuralGrokConfig,
    init_neuralgrok_state, neuralgrok_step,
)

__all__ = [
    # SuperGrok v2 core
    "PerParamState", "SuperGrok2State", "OptimizerConfig",
    "MetaNetWeights", "MambaScanWeights", "GRUWeights",
    "init_state", "init_per_param_state", "init_meta_weights",
    "supergrok2_step", "meta_net_forward",
    "mamba3_scan", "mini_gru", "peer_expert_forward",
    "bilevel_step",
    "create_mesh", "shard_params", "replicate_meta_weights",
    "pytorch_weights_to_jax", "jax_weights_to_pytorch",
    # Simple optimizers
    "GrokAdamWState", "GrokAdamWConfig", "init_grokadamw_state", "grokadamw_step",
    "LionState", "LionConfig", "init_lion_state", "lion_step",
    "GrokfastState", "GrokfastConfig", "init_grokfast_state", "grokfast_amplify",
    "ProdigyState", "ProdigyConfig", "init_prodigy_state", "prodigy_step",
    "MuonState", "MuonConfig", "init_muon_state", "muon_step",
    "LookSAMState", "LookSAMConfig", "init_looksam_state",
    "looksam_perturb", "looksam_compute_direction", "looksam_adjust_grad",
    "looksam_adam_step",
    # Meta-net optimizers
    "SuperGrok15State", "SuperGrok15Weights", "SuperGrok15Config",
    "init_supergrok15_state", "supergrok15_step",
    "SuperGrok11State", "SuperGrok11Weights", "SuperGrok11Config",
    "init_supergrok11_state", "supergrok11_step",
    "NeuralGrokState", "NeuralGrokWeights", "NeuralGrokConfig",
    "init_neuralgrok_state", "neuralgrok_step",
]
