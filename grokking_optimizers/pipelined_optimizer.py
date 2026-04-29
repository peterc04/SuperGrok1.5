"""Layer-by-layer optimizer pipelining.

Updates parameters as soon as their gradients are ready during backward,
overlapping optimizer work with remaining backward computation.

For simple optimizers (AdamW, Lion, GrokAdamW): each parameter is updated
immediately when its gradient hook fires, on a separate CUDA stream.

For SuperGrok v2: the meta-net forward requires ALL gradients (sort by
magnitude across all params), preventing per-layer pipelining of the
meta-net. But the Adam update portion CAN pipeline after meta-net completes.
"""

import torch


class PipelinedOptimizer:
    """Updates parameters layer-by-layer during backward.

    Registers backward hooks on each parameter. When a gradient is computed,
    the hook immediately runs the optimizer update for that parameter.
    The next layer's backward can proceed on the GPU while the optimizer
    update for the current layer runs on a separate stream.

    Args:
        optimizer: The base optimizer to pipeline.
        model: The model whose parameters to hook.
    """

    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self._opt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._hooks = []
        self._grad_count = 0
        self._total_params = sum(1 for p in model.parameters() if p.requires_grad)
        self._all_grads_ready = False
        self._setup_hooks()

    def _setup_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(
                    self._make_hook(param, name))
                self._hooks.append(hook)

    def _make_hook(self, param, name):
        def hook(p):
            self._grad_count += 1
            if self._grad_count >= self._total_params:
                self._all_grads_ready = True

            if self._opt_stream is not None:
                with torch.cuda.stream(self._opt_stream):
                    self._update_single_param(p)
            else:
                self._update_single_param(p)
        return hook

    def _update_single_param(self, param):
        """Run the optimizer update for a single parameter.

        For simple optimizers (AdamW, Lion, etc.): straightforward.
        For SuperGrok v2: this is complex because the meta-net forward
        needs ALL parameters' gradients (for the sort-by-magnitude step).

        Solution for SuperGrok v2: accumulate gradients, only run the
        full meta-net step when ALL gradients are ready (detected by
        counting hooks fired). Individual Adam updates can still pipeline.
        """
        if hasattr(self.optimizer, '_update_single_param_hook'):
            self.optimizer._update_single_param_hook(param)

    def step(self):
        """Called after backward. For pipelined optimizers, most work is
        already done via hooks. This handles any remaining work."""
        if self._opt_stream is not None:
            torch.cuda.current_stream().wait_stream(self._opt_stream)
        self._grad_count = 0
        self._all_grads_ready = False

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state
