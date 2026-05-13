"""Gradient compression for distributed training.

Implements:
  - INT8 compression: 4x bandwidth reduction, <0.1% accuracy loss
  - PowerSGD (rank-1 approximation): 10-100x reduction for large matrices
  - Error feedback: accumulate compression error for next step
"""

import torch
import torch.distributed as dist


class INT8GradientCompressor:
    """Compress gradients to INT8 before all-reduce."""

    def __init__(self):
        self.error_feedback = {}

    def compress(self, grad, param_id):
        """Compress FP32 gradient to INT8 + scale."""
        if param_id in self.error_feedback:
            grad = grad + self.error_feedback[param_id]

        scale = grad.abs().max() / 127.0
        if scale.item() < 1e-12:
            scale = torch.tensor(1e-12, device=grad.device)

        grad_int8 = (grad / scale).round().clamp(-127, 127).to(torch.int8)

        self.error_feedback[param_id] = grad - grad_int8.float() * scale

        return grad_int8, scale

    def decompress(self, grad_int8, scale):
        """Decompress INT8 gradient back to FP32."""
        return grad_int8.float() * scale


class PowerSGDCompressor:
    """PowerSGD low-rank gradient compression.

    Approximates gradient matrix G ≈ P @ Q^T where P is [m, rank] and Q is [n, rank].
    Communicates P and Q instead of G: (m+n)*rank instead of m*n.
    """

    def __init__(self, rank=1):
        self.rank = rank
        self.Q = {}

    def compress(self, grad, param_id):
        """Compress 2D gradient via PowerSGD."""
        if grad.dim() < 2:
            return grad, None

        m, n = grad.shape[0], grad.numel() // grad.shape[0]
        G = grad.view(m, n)

        if param_id not in self.Q:
            self.Q[param_id] = torch.randn(n, self.rank, device=grad.device)

        Q = self.Q[param_id]

        P = G @ Q
        P, _ = torch.linalg.qr(P)

        Q_new = G.t() @ P

        self.Q[param_id] = Q_new

        return (P, Q_new), grad.shape

    def decompress(self, compressed, shape):
        """Decompress PowerSGD: G ≈ P @ Q^T."""
        if compressed is None or not isinstance(compressed, tuple):
            return compressed
        P, Q = compressed
        return (P @ Q.t()).view(shape)
