"""Strict ops loader — no fallbacks.

Loads the per-arch specialized C++ extension. Raises RuntimeError if the
extension is not built or if the host arch is not in the supported set
{sm_80, sm_90, sm_100, gfx942}.

The _python_fallback module is ONLY for unit testing (imported explicitly
by tests). It is not a runtime fallback path.
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
            "SuperGrok C++ extension not built. "
            "Run: pip install -e . "
            "Supported arches: sm_80, sm_90, sm_100 (NVIDIA), gfx942 (AMD). "
            "CPU build is for testing only -- not a runtime fallback. "
            f"Original error: {e}"
        ) from e
