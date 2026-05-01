"""Fused (model, optimizer, arch) kernel dispatch.

Routes to the appropriate fused kernel based on detected hardware.
Falls back to separate forward/backward/step when fused kernel unavailable.
"""

from .dispatch import detect_arch, UnsupportedArchError

MODELS = ("transformer", "vit", "mamba")
OPTIMIZERS = ("grokadamw", "grokfast", "lion", "looksam", "moe", "muon",
              "neuralgrok", "prodigy", "supergrok2", "supergrok15", "supergrok11")

_FUSED_REGISTRY = {}  # (model, optimizer, arch) -> callable


def register_fused(model: str, optimizer: str, arch):
    """Decorator to register a fused kernel implementation."""
    def decorator(fn):
        _FUSED_REGISTRY[(model, optimizer, arch)] = fn
        return fn
    return decorator


def has_fused(model: str, optimizer: str, arch=None) -> bool:
    """Check if a fused kernel is available for the given triple."""
    if arch is None:
        arch = detect_arch()
    return (model, optimizer, arch) in _FUSED_REGISTRY


def dispatch_fused(model: str, optimizer: str, params, inputs, grads, state, lr, arch=None):
    """Dispatch to fused (model, optimizer, arch) kernel if available.

    Returns (output, new_state) or raises KeyError if no fused kernel registered.
    """
    if arch is None:
        arch = detect_arch()
    key = (model, optimizer, arch)
    if key not in _FUSED_REGISTRY:
        raise KeyError(f"No fused kernel for {key}. Available: {list(_FUSED_REGISTRY.keys())}")
    return _FUSED_REGISTRY[key](params, inputs, grads, state, lr)
