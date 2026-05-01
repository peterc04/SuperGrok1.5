"""NeuralGrok device-function template for TPU v5p (128-wide MXU).

NeuralGrok's MLP amplifier + Adam step as pure JAX ops.
The small MLP (1->H->1) maps well to XLA's vectorized operations.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def neuralgrok_amplifier(grad, W1, b1, W2, b2, alpha, beta):
    """Per-element MLP: Linear(1,H) -> ReLU -> Linear(H,1) -> scale.

    amplified = grad * (alpha * mlp_out + beta)
    """
    # W1: [H], b1: [H], W2: [H], b2: scalar
    hidden = jax.nn.relu(W1 * grad[..., None] + b1)  # [N, H]
    mlp_out = jnp.sum(W2 * hidden, axis=-1) + b2     # [N]
    return grad * (alpha * mlp_out + beta)


def neuralgrok_adam_step(param, exp_avg, exp_avg_sq, amplified_grad,
                         beta1, beta2, lr, weight_decay, eps, bc1, bc2):
    """Standard Adam update with amplified gradients."""
    ea = beta1 * exp_avg + (1.0 - beta1) * amplified_grad
    easq = beta2 * exp_avg_sq + (1.0 - beta2) * amplified_grad ** 2
    step_size = lr / bc1
    denom = jnp.sqrt(easq / bc2) + eps
    param = param * (1.0 - lr * weight_decay) - step_size * ea / denom
    return param, ea, easq
