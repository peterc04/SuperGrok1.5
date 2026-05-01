"""SuperGrok v1.5 device-function template for TPU v5p (128-wide MXU).

SG15 operations as pure JAX ops. The meta-net MLP(2->H->1) uses
JAX's vectorized operations which compile to efficient XLA HLO.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def sg15_mu_metanet(mu, grad, sharp, alpha, W1, b1, W2, b2, rescale):
    """EMA update + meta-net inference.

    mu = alpha * mu + (1-alpha) * grad
    MLP: Linear(2,H) -> GELU -> Linear(H,1)
    smart_grad = grad + rescale * mlp_out
    """
    mu = alpha * mu + (1.0 - alpha) * grad
    inputs = jnp.stack([grad, sharp], axis=-1)
    hidden = jax.nn.gelu(jnp.einsum('...i,hi->...h', inputs, W1) + b1)
    mlp_out = jnp.sum(W2 * hidden, axis=-1) + b2
    smart_grad = grad + rescale * mlp_out
    return mu, smart_grad


def sg15_adam_decay(param, exp_avg, exp_avg_sq, smart_grad, normal_grad,
                    gate_signal, beta1, beta2, lr, weight_decay, eps, bc1, bc2):
    """Fused gating + Adam + progressive weight decay.

    Blends smart_grad with normal_grad via sigmoid gate.
    """
    gate = jax.nn.sigmoid(gate_signal)
    blended = gate * smart_grad + (1.0 - gate) * normal_grad
    ea = beta1 * exp_avg + (1.0 - beta1) * blended
    easq = beta2 * exp_avg_sq + (1.0 - beta2) * blended ** 2
    step_size = lr / bc1
    denom = jnp.sqrt(easq / bc2) + eps
    param = param * (1.0 - lr * weight_decay) - step_size * ea / denom
    return param, ea, easq
