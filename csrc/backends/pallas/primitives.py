"""Pallas / JAX primitives — shared across all 11 launch_*.py files.

Vendor-specific helpers for TPU v5p (128-wide MXU). The optimizer math
itself is identical to the C++ algorithm headers in csrc/algorithms/;
this file provides:

  - JAX/Pallas wrappers for common operations
  - Detection helpers (TPU version, lane width)
  - Functional state init helpers
  - Adam/EMA building blocks expressed as pure JAX
  - Re-exports of the working Pallas kernels in csrc/kernels/tpu/

The 11 launch_<optimizer>.py files import from this module and the
corresponding algorithm header is referenced in comments only (it's
a C++ header and not importable from Python).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


# ---------------------------------------------------------------------------
# TPU version detection
# ---------------------------------------------------------------------------


def detect_tpu_version() -> str:
    """Return the TPU generation string ("v4", "v5e", "v5p", "v6e") or "cpu".

    Falls back to "v4" sentinel if detection fails (matches existing
    csrc/kernels/tpu/_pallas_kernels.py convention).
    """
    try:
        devs = jax.devices()
        if not devs:
            return "cpu"
        kind = getattr(devs[0], "device_kind", "") or ""
        kind = kind.lower()
        for v in ("v6e", "v5p", "v5e", "v4"):
            if v in kind:
                return v
        return "v4"  # sentinel
    except Exception:
        return "v4"


def mxu_tile_width() -> int:
    """MXU lane width for the active TPU. 128 for v4/v5e/v5p, 256 for v6e."""
    v = detect_tpu_version()
    return 256 if v == "v6e" else 128


# ---------------------------------------------------------------------------
# Functional optimizer building blocks (pure JAX, no Pallas).
# ---------------------------------------------------------------------------


def ema_update(state: jnp.ndarray, g: jnp.ndarray, decay: float) -> jnp.ndarray:
    """state' = decay * state + (1 - decay) * g."""
    return decay * state + (1.0 - decay) * g


def ema_sq_update(state: jnp.ndarray, g: jnp.ndarray, decay: float) -> jnp.ndarray:
    """state' = decay * state + (1 - decay) * g^2."""
    return decay * state + (1.0 - decay) * g * g


def adamw_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """AdamW per-tensor step (vectorized over all elements)."""
    m = ema_update(exp_avg, grad, beta1)
    v = ema_sq_update(exp_avg_sq, grad, beta2)
    m_hat = m * bc1
    v_hat = v * bc2
    update = m_hat / (jnp.sqrt(v_hat) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v


def lion_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lion per-tensor step. Sign-based, single-state."""
    interp = beta1 * exp_avg + (1.0 - beta1) * grad
    update = jnp.sign(interp)
    new_param = param - lr * (update + wd * param)
    new_exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    return new_param, new_exp_avg


# ---------------------------------------------------------------------------
# Re-exports of working Pallas kernels (lives in csrc/kernels/tpu/).
# After Phase 7, these will be moved to csrc/backends/pallas/.
# ---------------------------------------------------------------------------

try:
    from csrc.kernels.tpu._pallas_kernels import (
        mamba3_scan_pallas_tile128,
        mamba3_scan_pallas_tile256,
        pallas_persistent_scan_fused_elem_tile128,
        pallas_persistent_scan_fused_elem_tile256,
        pallas_fused_gru_peer,
        tpu_auto_select_scan,
        tpu_auto_select_fused_scan_elem,
    )
    _PALLAS_AVAILABLE = True
except ImportError:
    _PALLAS_AVAILABLE = False


def pallas_available() -> bool:
    return _PALLAS_AVAILABLE
