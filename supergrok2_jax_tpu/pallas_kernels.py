"""
Pallas Custom Kernels for SuperGrok v2 (JAX)

Design decision: Start WITHOUT Pallas. Use pure JAX everywhere.
Only add Pallas kernels for operations where profiling shows XLA's
auto-compilation is measurably slower than a hand-written kernel.

Primary candidates for Pallas optimization (if needed):
  1. The sort operation: jnp.argsort may not be optimal on TPU
  2. Expert weight gather: expert_W1[expert_idx] scatter-gather
  3. Fused GRU+PEER: single kernel for combined computation

Current status: NO Pallas kernels implemented. Pure JAX is used for
all operations. Profile on TPU before adding Pallas.

To add a Pallas kernel:
  1. Profile the pure JAX implementation on TPU
  2. Identify the bottleneck (must be >20% of step time)
  3. Write the Pallas kernel
  4. Verify output matches pure JAX within 1e-5
  5. Benchmark: Pallas must be >1.5x faster to justify complexity
"""

# Placeholder — Pallas kernels will be added here if profiling
# shows a clear bottleneck in the pure JAX implementation.
#
# Example (not yet implemented):
#
# try:
#     from jax.experimental import pallas as pl
#     from jax.experimental.pallas import tpu as pltpu
#     _HAS_PALLAS = True
# except ImportError:
#     _HAS_PALLAS = False
#
# def bitonic_sort_by_key(keys, values):
#     """Pallas bitonic sort (if needed for TPU performance)."""
#     raise NotImplementedError("Profile first, optimize second")
