import torch

class PartialGraphOptimizer:
    """Wraps an optimizer to use CUDA graph capture for the step() call.

    The first call to step() is a warmup (no graph). The second call captures
    the step into a CUDA graph. Subsequent calls replay the graph.

    This eliminates kernel launch overhead for optimizers with many small kernels.
    Invalidation: call invalidate() after param group changes or lr schedule steps
    that modify optimizer state shape.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._graph = None
        self._warmup_done = False
        self._stream = None

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("PartialGraphOptimizer does not support closures")

        if not self._warmup_done:
            # Warmup: run step normally
            self.optimizer.step()
            self._warmup_done = True
            return

        if self._graph is None:
            # Capture
            self._stream = torch.cuda.Stream()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                # Warmup in capture stream
                self.optimizer.step()
            # Now capture
            with torch.cuda.graph(self._graph, stream=self._stream):
                self.optimizer.step()
            return

        # Replay
        self._graph.replay()

    def invalidate(self):
        """Call after lr schedule changes or param group modifications."""
        if self._graph is not None:
            del self._graph
            self._graph = None
            self._warmup_done = False

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.invalidate()
        self.optimizer.load_state_dict(state_dict)
