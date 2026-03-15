"""
PyTorch <-> JAX Bridge for SuperGrok v2

Provides conversion utilities between PyTorch meta-net weights and
JAX MetaNetWeights. Useful for:
  1. Migrating trained models from PyTorch to JAX
  2. Cross-framework testing (export test vectors)
  3. Using JAX optimizer with a PyTorch-trained meta-net

Usage::

    # PyTorch -> JAX
    jax_weights = pytorch_weights_to_jax(pytorch_meta_net)

    # JAX -> PyTorch
    jax_weights_to_pytorch(jax_weights, pytorch_meta_net)

    # Export test vectors for cross-framework testing
    vectors = export_test_vectors(pytorch_meta_net, grad, sharpness, ...)
"""

import jax.numpy as jnp
from typing import Any, Dict, Optional

from .scan import MambaScanWeights
from .gru import GRUWeights
from .mamba3_peer_metanet_jax import MetaNetWeights


def _to_jnp(tensor) -> jnp.ndarray:
    """Convert a PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().float().numpy())


def _to_torch(array, device='cpu'):
    """Convert a JAX array to PyTorch tensor."""
    import torch
    return torch.from_numpy(array.__array__()).to(device)


def pytorch_weights_to_jax(meta_net_pytorch) -> MetaNetWeights:
    """Convert PyTorch Mamba3PEERMetaNet to JAX MetaNetWeights.

    Args:
        meta_net_pytorch: PyTorch Mamba3PEERMetaNet instance

    Returns:
        MetaNetWeights (JAX)
    """
    mn = meta_net_pytorch

    mamba_fwd = MambaScanWeights(
        in_proj_W=_to_jnp(mn.mamba_fwd.in_proj.weight),
        dt_proj_W=_to_jnp(mn.mamba_fwd.dt_proj.weight),
        dt_proj_b=_to_jnp(mn.mamba_fwd.dt_proj.bias),
        B_proj_W=_to_jnp(mn.mamba_fwd.B_proj.weight),
        C_proj_W=_to_jnp(mn.mamba_fwd.C_proj.weight),
        A_log=_to_jnp(mn.mamba_fwd.A_log),
        D=_to_jnp(mn.mamba_fwd.D),
        rope_freq=_to_jnp(mn.mamba_fwd.rope_freq),
        out_proj_W=_to_jnp(mn.mamba_fwd.out_proj.weight),
    )

    mamba_bwd = MambaScanWeights(
        in_proj_W=_to_jnp(mn.mamba_bwd.in_proj.weight),
        dt_proj_W=_to_jnp(mn.mamba_bwd.dt_proj.weight),
        dt_proj_b=_to_jnp(mn.mamba_bwd.dt_proj.bias),
        B_proj_W=_to_jnp(mn.mamba_bwd.B_proj.weight),
        C_proj_W=_to_jnp(mn.mamba_bwd.C_proj.weight),
        A_log=_to_jnp(mn.mamba_bwd.A_log),
        D=_to_jnp(mn.mamba_bwd.D),
        rope_freq=_to_jnp(mn.mamba_bwd.rope_freq),
        out_proj_W=_to_jnp(mn.mamba_bwd.out_proj.weight),
    )

    gru = GRUWeights(
        Wz=_to_jnp(mn.gru.W_z.weight),
        bz=_to_jnp(mn.gru.W_z.bias),
        Wr=_to_jnp(mn.gru.W_r.weight),
        br=_to_jnp(mn.gru.W_r.bias),
        Wh=_to_jnp(mn.gru.W_h.weight),
        bh=_to_jnp(mn.gru.W_h.bias),
    )

    peer_query_Ws = jnp.stack([
        _to_jnp(q.weight) for q in mn.peer_queries
    ])
    prod_keys_A = jnp.stack([
        _to_jnp(k) for k in mn.product_keys_A
    ])
    prod_keys_B = jnp.stack([
        _to_jnp(k) for k in mn.product_keys_B
    ])

    return MetaNetWeights(
        input_proj_W=_to_jnp(mn.input_proj.weight),
        input_proj_b=_to_jnp(mn.input_proj.bias),
        mamba_fwd=mamba_fwd,
        mamba_bwd=mamba_bwd,
        gru=gru,
        peer_query_Ws=peer_query_Ws,
        prod_keys_A=prod_keys_A,
        prod_keys_B=prod_keys_B,
        expert_W1=_to_jnp(mn.expert_W1),
        expert_b1=_to_jnp(mn.expert_b1),
        expert_W2=_to_jnp(mn.expert_W2),
        expert_b2=_to_jnp(mn.expert_b2),
        rescale=jnp.array(mn.rescale),
        expert_counts=_to_jnp(mn.expert_counts),
    )


def jax_weights_to_pytorch(
    jax_weights: MetaNetWeights,
    meta_net_pytorch,
    device: str = 'cpu',
) -> None:
    """Copy JAX MetaNetWeights back to PyTorch Mamba3PEERMetaNet.

    Args:
        jax_weights: JAX MetaNetWeights
        meta_net_pytorch: PyTorch Mamba3PEERMetaNet instance (modified in place)
        device: target PyTorch device
    """
    mn = meta_net_pytorch

    mn.input_proj.weight.data = _to_torch(jax_weights.input_proj_W, device)
    mn.input_proj.bias.data = _to_torch(jax_weights.input_proj_b, device)

    # Mamba forward
    mn.mamba_fwd.in_proj.weight.data = _to_torch(jax_weights.mamba_fwd.in_proj_W, device)
    mn.mamba_fwd.dt_proj.weight.data = _to_torch(jax_weights.mamba_fwd.dt_proj_W, device)
    mn.mamba_fwd.dt_proj.bias.data = _to_torch(jax_weights.mamba_fwd.dt_proj_b, device)
    mn.mamba_fwd.B_proj.weight.data = _to_torch(jax_weights.mamba_fwd.B_proj_W, device)
    mn.mamba_fwd.C_proj.weight.data = _to_torch(jax_weights.mamba_fwd.C_proj_W, device)
    mn.mamba_fwd.A_log.data = _to_torch(jax_weights.mamba_fwd.A_log, device)
    mn.mamba_fwd.D.data = _to_torch(jax_weights.mamba_fwd.D, device)
    mn.mamba_fwd.rope_freq.data = _to_torch(jax_weights.mamba_fwd.rope_freq, device)
    mn.mamba_fwd.out_proj.weight.data = _to_torch(jax_weights.mamba_fwd.out_proj_W, device)

    # Mamba backward
    mn.mamba_bwd.in_proj.weight.data = _to_torch(jax_weights.mamba_bwd.in_proj_W, device)
    mn.mamba_bwd.dt_proj.weight.data = _to_torch(jax_weights.mamba_bwd.dt_proj_W, device)
    mn.mamba_bwd.dt_proj.bias.data = _to_torch(jax_weights.mamba_bwd.dt_proj_b, device)
    mn.mamba_bwd.B_proj.weight.data = _to_torch(jax_weights.mamba_bwd.B_proj_W, device)
    mn.mamba_bwd.C_proj.weight.data = _to_torch(jax_weights.mamba_bwd.C_proj_W, device)
    mn.mamba_bwd.A_log.data = _to_torch(jax_weights.mamba_bwd.A_log, device)
    mn.mamba_bwd.D.data = _to_torch(jax_weights.mamba_bwd.D, device)
    mn.mamba_bwd.rope_freq.data = _to_torch(jax_weights.mamba_bwd.rope_freq, device)
    mn.mamba_bwd.out_proj.weight.data = _to_torch(jax_weights.mamba_bwd.out_proj_W, device)

    # GRU
    mn.gru.W_z.weight.data = _to_torch(jax_weights.gru.Wz, device)
    mn.gru.W_z.bias.data = _to_torch(jax_weights.gru.bz, device)
    mn.gru.W_r.weight.data = _to_torch(jax_weights.gru.Wr, device)
    mn.gru.W_r.bias.data = _to_torch(jax_weights.gru.br, device)
    mn.gru.W_h.weight.data = _to_torch(jax_weights.gru.Wh, device)
    mn.gru.W_h.bias.data = _to_torch(jax_weights.gru.bh, device)

    # PEER
    for h in range(len(mn.peer_queries)):
        mn.peer_queries[h].weight.data = _to_torch(jax_weights.peer_query_Ws[h], device)
        mn.product_keys_A[h].data = _to_torch(jax_weights.prod_keys_A[h], device)
        mn.product_keys_B[h].data = _to_torch(jax_weights.prod_keys_B[h], device)

    # Experts
    mn.expert_W1.data = _to_torch(jax_weights.expert_W1, device)
    mn.expert_b1.data = _to_torch(jax_weights.expert_b1, device)
    mn.expert_W2.data = _to_torch(jax_weights.expert_W2, device)
    mn.expert_b2.data = _to_torch(jax_weights.expert_b2, device)


def export_test_vectors(
    meta_net_pytorch=None,
    grad=None,
    sharpness=None,
    gru_state=None,
    mamba_fwd_state=None,
    mamba_bwd_state=None,
    save_path=None,
) -> Dict:
    """Export input/output pairs for cross-framework testing.

    Two modes:
      1. With meta_net_pytorch + tensors: runs PyTorch forward and captures I/O
      2. With save_path only: generates deterministic test vectors and saves to .npz

    Args:
        meta_net_pytorch: PyTorch Mamba3PEERMetaNet (mode 1)
        grad: [N] tensor (mode 1)
        sharpness: [N] tensor (mode 1)
        gru_state: [N, gru_hidden] tensor (mode 1)
        mamba_fwd_state: [d_inner, d_state] tensor (mode 1)
        mamba_bwd_state: [d_inner, d_state] tensor (mode 1)
        save_path: if provided, save vectors to this .npz path (mode 2)

    Returns:
        dict with 'inputs' and 'outputs' sub-dicts of numpy arrays
    """
    import numpy as np

    # Mode 2: standalone test vector generation (no PyTorch needed)
    if meta_net_pytorch is None and save_path is not None:
        rng = np.random.RandomState(42)
        N = 64
        d_model, d_state, d_inner, gru_hidden = 8, 16, 16, 4

        vectors = {
            'inputs': {
                'grad': rng.randn(N).astype(np.float32) * 0.1,
                'sharpness': np.abs(rng.randn(N).astype(np.float32)) * 0.01,
                'gru_state': np.zeros((N, gru_hidden), dtype=np.float32),
                'mamba_fwd_state': np.zeros((d_inner, d_state), dtype=np.float32),
                'mamba_bwd_state': np.zeros((d_inner, d_state), dtype=np.float32),
            },
            'config': {
                'N': N,
                'd_model': d_model,
                'd_state': d_state,
                'd_inner': d_inner,
                'gru_hidden': gru_hidden,
            },
        }

        # Flatten for npz save
        flat = {}
        for section in vectors:
            for key, val in vectors[section].items():
                flat[f"{section}/{key}"] = np.array(val) if not isinstance(val, np.ndarray) else val

        np.savez(save_path, **flat)
        return vectors

    # Mode 1: PyTorch forward pass capture
    import torch

    with torch.no_grad():
        smart_grad, new_gru, new_fwd, new_bwd = meta_net_pytorch.forward_for_bilevel(
            grad, sharpness, gru_state, mamba_fwd_state, mamba_bwd_state)

    result = {
        'inputs': {
            'grad': grad.cpu().numpy(),
            'sharpness': sharpness.cpu().numpy(),
            'gru_state': gru_state.cpu().numpy(),
            'mamba_fwd_state': mamba_fwd_state.cpu().numpy(),
            'mamba_bwd_state': mamba_bwd_state.cpu().numpy(),
        },
        'outputs': {
            'smart_grad': smart_grad.detach().cpu().numpy(),
            'new_gru_state': new_gru.detach().cpu().numpy(),
            'new_mamba_fwd_state': new_fwd.detach().cpu().numpy(),
            'new_mamba_bwd_state': new_bwd.detach().cpu().numpy(),
        },
    }

    if save_path is not None:
        import numpy as np
        flat = {}
        for section in result:
            for key, val in result[section].items():
                flat[f"{section}/{key}"] = val
        np.savez(save_path, **flat)

    return result
