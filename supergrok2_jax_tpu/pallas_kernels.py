"""
Pallas Custom Kernels for SuperGrok v2 (JAX)

Design decision: Start WITHOUT Pallas. Use pure JAX everywhere.
Only add Pallas kernels for operations where profiling shows XLA's
auto-compilation is measurably slower than a hand-written kernel.

Primary candidates for Pallas optimization (if needed):
  1. The sort operation: jnp.argsort may not be optimal on TPU
  2. Expert weight gather: expert_W1[expert_idx] scatter-gather
  3. Fused GRU+PEER: single kernel for combined computation

Decision criteria (all must be met):
  1. Profiling identifies a bottleneck: >20% of optimizer step time
  2. Pallas kernel is >1.5x faster than pure JAX
  3. Output matches pure JAX within 1e-5

Current status: NO Pallas kernels active. Pure JAX is used for all
operations. Profile on TPU before adding Pallas.
"""

# ── Pallas availability check ────────────────────────────────────────
_HAS_PALLAS = False
try:
    from jax.experimental import pallas as pl       # noqa: F401
    from jax.experimental.pallas import tpu as pltpu  # noqa: F401
    _HAS_PALLAS = True
except ImportError:
    pass


def pallas_mamba3_scan(
    x_sorted,
    weights,
    initial_state=None,
    reverse=False,
):
    """Pallas-accelerated Mamba-3 scan (stub — falls back to pure JAX).

    This function exists as a drop-in replacement for scan.mamba3_scan.
    Currently it always falls back to the pure JAX implementation.

    To activate a Pallas kernel:
      1. Profile mamba3_scan on your target TPU (v4/v5)
      2. If scan is >20% of step time, implement the kernel below
      3. Verify output matches pure JAX within 1e-5
      4. Benchmark: Pallas must be >1.5x faster
      5. Set _USE_PALLAS_SCAN = True

    Args:
        x_sorted: [N, d_model] sorted input features
        weights: MambaScanWeights namedtuple
        initial_state: [d_inner, d_state] or None
        reverse: if True, scan in reverse direction

    Returns:
        output: [N, d_model] scan output
        final_state: [d_inner, d_state]
    """
    # Always fall back to pure JAX — Pallas not yet justified by profiling
    from .scan import mamba3_scan
    return mamba3_scan(x_sorted, weights, initial_state, reverse)


# ── Future Pallas kernel template (commented out) ────────────────────
#
# _USE_PALLAS_SCAN = False
#
# if _HAS_PALLAS and _USE_PALLAS_SCAN:
#     def _pallas_scan_kernel(
#         x_ref, dt_ref, B_ref, C_ref, A_bar_ref, D_ref,
#         z_ref, h_ref, out_ref,
#         *,
#         d_inner, d_state, block_n,
#     ):
#         """Pallas kernel for Mamba-3 scan on TPU.
#
#         Uses Megacore tiling for the scan's sequential dependency.
#         The key insight: while the scan is inherently sequential across N,
#         the d_inner and d_state dimensions can be parallelized across
#         TPU's 128x128 MXU tiles.
#         """
#         # TODO: implement when profiling justifies it
#         raise NotImplementedError("Profile first, optimize second")
