"""Pure-Python AdamW step used by optimizers that need a standard Adam update.

PyTorch's built-in fused AdamW handles the CUDA optimization internally,
so a custom C++ kernel is unnecessary.  This helper provides the same
numerics as the standard AdamW algorithm with decoupled weight decay.
"""

import torch


@torch.no_grad()
def adamw_step(params, grads, exp_avgs, exp_avg_sqs, steps, lr, beta1, beta2, eps, wd):
    """Single AdamW step over a list of parameters."""
    for p, g, ea, easq, step in zip(params, grads, exp_avgs, exp_avg_sqs, steps):
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        ea.mul_(beta1).add_(g, alpha=1 - beta1)
        easq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        step_size = lr / bc1
        denom = (easq / bc2).sqrt().add_(eps)
        p.mul_(1 - lr * wd)
        p.addcdiv_(ea, denom, value=-step_size)
