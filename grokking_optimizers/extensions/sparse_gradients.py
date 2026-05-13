import torch

class SparseGradientHandler:
    """Converts sparse gradients to dense before optimizer step.

    Some embedding layers produce sparse gradients (torch.sparse_coo).
    Most custom CUDA optimizers require dense gradients. This handler
    densifies sparse gradients in-place before the optimizer step.

    Usage:
        handler = SparseGradientHandler(optimizer)
        # In training loop:
        handler.densify_gradients()
        optimizer.step()
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def densify_gradients(self):
        """Convert any sparse gradients to dense, in-place."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and p.grad.is_sparse:
                    p.grad = p.grad.to_dense()

    def has_sparse_gradients(self):
        """Check if any parameters currently have sparse gradients."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and p.grad.is_sparse:
                    return True
        return False
