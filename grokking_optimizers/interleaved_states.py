"""Memory-optimized optimizer state storage.

Interleaves multiple state tensors into a single allocation
for cache-friendly access patterns when N is small enough
to fit in L2 cache.
"""

import torch


class InterleavedStates:
    """Interleaved optimizer state storage.

    For small parameter tensors (N < 65536), interleaving state tensors
    improves L2 cache hit rate by keeping exp_avg[i], exp_avg_sq[i], mu[i]
    in the same cache line.

    For large parameter tensors, separate allocations allow independent
    prefetch streams, which is more efficient.
    """

    INTERLEAVE_THRESHOLD = 65536

    def __init__(self, param, num_states):
        self.N = param.numel()
        self.num_states = num_states
        self.N_padded = ((self.N + 3) // 4) * 4
        self.use_interleaved = (self.N < self.INTERLEAVE_THRESHOLD)

        if self.use_interleaved:
            self.buffer = torch.zeros(
                self.N_padded * num_states,
                dtype=param.dtype, device=param.device)
        else:
            self._states = [
                torch.zeros(self.N, dtype=param.dtype, device=param.device)
                for _ in range(num_states)
            ]

    def get_state(self, idx):
        """Return a view of state idx."""
        if self.use_interleaved:
            return self.buffer[idx * self.N_padded:(idx + 1) * self.N_padded][:self.N]
        else:
            return self._states[idx]
