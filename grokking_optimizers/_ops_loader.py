"""Strict ops loader — no fallbacks.

If the C++ extension is not built, raises RuntimeError with instructions.
The _python_fallback module is ONLY for unit testing (imported explicitly by tests).
"""

_cached_ops = None


def get_ops():
    global _cached_ops
    if _cached_ops is not None:
        return _cached_ops

    try:
        from grokking_optimizers import _ops
        _cached_ops = _ops
        return _ops
    except ImportError as e:
        raise RuntimeError(
            "SuperGrok v2 C++ extension not built. "
            "Run: pip install -e . "
            "(requires CUDA toolkit or ROCm for GPU, or C++ compiler for CPU-only). "
            f"Original error: {e}"
        ) from e
