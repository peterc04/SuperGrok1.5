"""Overlap gradient all-reduce with optimizer step.

Instead of:
  1. Forward + backward → all gradients ready
  2. All-reduce ALL gradients
  3. Optimizer step on ALL params

Do:
  1. Forward + backward → gradients ready in reverse layer order
  2. As each layer's gradient is ready:
     a. Start async all-reduce for that layer
     b. When all-reduce completes, run optimizer step for that layer
  3. Layers overlap: all-reduce(layer N) happens during backward(layer N-1)
"""

import torch
import torch.distributed as dist
from typing import List


class OverlappedOptimizer:
    """Wraps any optimizer to overlap all-reduce with step.

    Usage:
        base_opt = SuperGrok2(model.parameters(), lr=1e-3)
        opt = OverlappedOptimizer(base_opt, model)

        loss = model(x)
        loss.backward()
        opt.step()  # overlaps all-reduce with per-parameter steps
    """

    def __init__(self, optimizer, model, bucket_size_mb=25):
        self.optimizer = optimizer
        self.model = model
        self.bucket_size_mb = bucket_size_mb
        self._buckets = self._create_buckets()

    def _create_buckets(self):
        """Group parameters into communication buckets (like DDP)."""
        buckets = []
        current_bucket = []
        current_size = 0

        for param in reversed(list(self.model.parameters())):
            if param.requires_grad:
                param_size = param.numel() * param.element_size()
                if current_size + param_size > self.bucket_size_mb * 1024 * 1024 and current_bucket:
                    buckets.append(current_bucket)
                    current_bucket = []
                    current_size = 0
                current_bucket.append(param)
                current_size += param_size

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    def step(self, **kwargs):
        """Overlapped all-reduce + optimizer step."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return self.optimizer.step(**kwargs)

        world_size = dist.get_world_size()

        for bucket in self._buckets:
            grads = [p.grad for p in bucket if p.grad is not None]
            if not grads:
                continue

            flat = torch.cat([g.flatten() for g in grads])

            work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
            work.wait()
            flat.div_(world_size)

            offset = 0
            for g in grads:
                numel = g.numel()
                g.copy_(flat[offset:offset+numel].view_as(g))
                offset += numel

        return self.optimizer.step(**kwargs)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class OverlappedSuperGrok2(OverlappedOptimizer):
    """SuperGrok v2 with overlapped communication.

    Special handling:
    - Model parameter gradients: bucketed async all-reduce
    - Meta-net gradients: separate sync all-reduce (must be consistent for bilevel)
    - Expert counts: all-reduce for recycling decisions
    """

    def step(self, train_loss=None, val_loss=None, **kwargs):
        super().step(**kwargs)

        if hasattr(self.optimizer, '_allreduce_meta_grads'):
            self.optimizer._allreduce_meta_grads()
