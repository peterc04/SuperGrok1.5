"""tests/tpu — TPU v6e (Pallas/JAX) pre-silicon validation.

CPU-only suites that exercise the Pallas optimizer kernels in
``csrc/backends/pallas/`` under ``pl.pallas_call(interpret=True)`` so that the
TPU-side math is functionally validated *before* any silicon exists. They are
skipped cleanly when JAX is not installed (``pytest.importorskip("jax")``), so
CI stays green on a torch-only runner.
"""
